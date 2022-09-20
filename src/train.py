import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

import src.utils as utils
from src.utils import sampling_assignment

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from ray import tune
import time
import os


# TODO: save only sum, not each value.
class Buffer():
    def __init__(self, batch_size, num_variables) -> None:
        # Episode Buffer
        self.action_logits = torch.empty(size=(batch_size, num_variables, 2))  # verbose 3
        # ::buffer_action_logits:: [batch_size, seq_len=num_variables, feature_size=2]
        self.action_softmax = torch.empty(size=(batch_size, num_variables, 2))  # verbose 3
        # ::buffer_action_softmax:: [batch_size, seq_len=num_variables, feature_size=2]
        self.action = torch.empty(size=(batch_size, num_variables), dtype = torch.int64)
        # ::buffer_action:: [batch_size, seq_len=num_variables]
        self.action_log_prob = torch.empty(size=(batch_size, num_variables))
        # ::buffer_action_log_prob:: [batch_size, seq_len=num_variables]
        self.entropy = torch.empty(size=(batch_size, num_variables))
        # ::buffer_entropy:: [batch_size, seq_len=num_variables]
    
    def update(self, idx, t, action_logits, action_softmax, action, action_log_prob, entropy):
        self.action_logits[idx, t] = action_logits.squeeze(1)  # Verbose 3
        self.action_softmax[idx, t] = action_softmax.squeeze(1)  # Verbose 3
        self.action[idx, t] = action.view(-1)
        self.action_log_prob[idx, t] = action_log_prob.view(-1)
        self.entropy[idx, t] = entropy.view(-1)


class Episode():
    def __init__(self,
                 formula,
                 num_variables,
                 variables,
                 policy_network,
                 device,
                 strategy = 'sampled',
                 batch_size = 1,
                 permute_vars = False,
                 permute_seed = None, #2147483647
                 baseline = None,
                 entropy_weight = 0):
    
        self.formula = formula
        self.num_variables = num_variables
        self.variables = variables
        self.policy_network = policy_network
        self.device = device
        self.strategy = strategy
        self.batch_size = batch_size
        self.permute_vars = permute_vars
        self.permute_seed = permute_seed
        self.baseline = baseline
        self.entropy_weight = entropy_weight

        #self.buffer = Buffer(batch_size, num_variables)


    def initialize(self):
        """
        Attributes
        ----------
            :: enc_output ::
            :: var ::
            :: init_action ::
            :: context ::
            :: state ::
            :: permutation ::
            :: buffer ::
        """
        # Encoder
        self.enc_output = None
        if self.policy_network.encoder is not None:
            self.enc_output = self.policy_network.encoder(self.formula, self.num_variables, self.variables)

        # Initialize Decoder Variables 
        self.var = self.policy_network.init_dec_var(self.enc_output, self.formula, self.num_variables, self.variables, self.batch_size)
        # ::var:: [batch_size, seq_len, feature_size]

        # Initialize action_prev at time t=0 with token 2.
        #   Token 0 is for assignment 0, token 1 for assignment 1
        self.init_action = torch.tensor([2] * self.batch_size, dtype=torch.long).reshape(-1,1,1).to(self.device)
        # ::action_prev:: [batch_size, seq_len=1, feature_size=1]

        # Initialize Decoder Context
        self.context = self.policy_network.init_dec_context(self.enc_output, self.formula, self.num_variables, self.variables, self.batch_size).to(self.device)
        # ::context:: [batch_size, feature_size]

        # Initialize Decoder state
        self.init_state = self.policy_network.init_dec_state(self.enc_output, self.batch_size)
        if self.init_state is not None: 
            self.init_state = self.init_state.to(self.device)
        # ::init_state:: [num_layers, batch_size, hidden_size]

        # Permutation of the indices of the variables
        if self.permute_vars:
            if self.permute_seed is not None:
                gen = torch.Generator(device=self.device)
                self.permutation = torch.cat([torch.randperm(self.num_variables, generator=gen.manual_seed(self.permute_seed)).unsqueeze(0) for _ in range(self.batch_size)], dim=0).permute(1,0)
            else:
                self.permutation = torch.cat([torch.randperm(self.num_variables).unsqueeze(0) for _ in range(self.batch_size)], dim=0).permute(1,0)
        else:
            self.permutation = torch.cat([torch.tensor([i for i in range(self.num_variables)]).unsqueeze(0) for _ in range(self.batch_size)], dim=0).permute(1,0)
        # ::permutation:: [num_variables, batch_size]; e.g.: [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]

        # Initialize buffer
        self.buffer = Buffer(self.batch_size, self.num_variables)


    def prediction(self):
        """
        Runs an episode and updates the buffer.
        """
        
        action_prev = self.init_action
        state = self.init_state
        batch_idx = [i for i in range(self.batch_size)]
        
        # Run an episode
        for var_idx in self.permutation:
            var_t = self.var[batch_idx, var_idx].unsqueeze(1).to(self.device)
            assert var_t.shape == (self.batch_size, 1, self.var.shape[-1])
            # ::var_t:: [batch_size, seq_len=1, feature_size]

            # Action logits
            action_logits, state = self.policy_network.decoder((var_t, action_prev, self.context), state)
            # ::action_logits:: [batch_size, seq_len=1, feature_size=2]

            # Prob distribution over actions
            action_softmax = F.softmax(action_logits, dim = -1)  # Verbose 3
            action_dist = distributions.Categorical(logits= action_logits)
            
            # Action selection
            if self.strategy == 'greedy':
                # Choose greedy action
                action = torch.argmax(action_logits, dim=-1)
            elif self.strategy == 'sampled':
                # Sample a rondom action 
                action = action_dist.sample()
            else:
                raise TypeError("{} is not a valid strategy, try with 'greedy' or 'sampled'.".format(self.strategy))
            
            # Log-prob of the action
            action_log_prob = action_dist.log_prob(action)

            # Computing Entropy
            entropy = action_dist.entropy()
            
            # Take the choosen action
            #-------

            # Update buffer
            self.buffer.update(batch_idx, var_idx, action_logits, action_softmax, action, action_log_prob, entropy)
            
            #actions_logits.append(list(np.around(action_logits.detach().cpu().numpy().flatten(), 2)))
            #actions_softmax.append(list(np.around(F.softmax(action_logits.detach(), -1).numpy().flatten(), 2)))
            #actions.append(action.item())
            #action_log_probs.append(action_log_prob)
            #entropies.append(action_dist_entropy)

            action_prev = action.unsqueeze(dim=-1)
            #::action_prev:: [batch_size, seq_len=1, feature_size=1]
        
        return self.buffer.action  # self.buffer.action.detach().cpu().numpy()


    def get_loss(self):
        """
        Attributes
        ----------
            :: mean_num_sat ::
            :: baseline_val ::
            :: mean_action_log_prob ::
            :: mean_entropy ::
            :: loss ::
        """
        is_training = self.policy_network.training
        
        self.policy_network.eval()
        with torch.no_grad():
            # Compute num of sat clauses (mean over batch)
            self.mean_num_sat = utils.mean_sat_clauses(self.formula, self.buffer.action.detach().cpu().numpy()).detach()

            # Compute baseline
            # TODO: Test baseline
            # TODO: Baseline runs in parallel rollout (by batch)
            self.baseline_val = torch.tensor(0, dtype=float).detach()
            if self.baseline is not None:
                self.baseline_val = self.baseline(self.formula, self.num_variables, self.variables, self.policy_network, self.device, self.permute_vars, self.permute_seed).detach()

        if is_training:
            self.policy_network.train()
        
        # Get loss (mean over the batch)
        total_action_log_prob = self.buffer.action_log_prob.sum(dim=-1)
        self.mean_action_log_prob =  total_action_log_prob.mean()  # mean over the batch
        total_entropy = self.buffer.entropy.sum(-1)
        self.mean_entropy = total_entropy.mean().detach()  # mean over the batch
        self.loss = - ((self.mean_num_sat - self.baseline_val) * self.mean_action_log_prob) - (self.entropy_weight * self.mean_entropy)
        # TODO: entropy detach?

        return self.loss, self.mean_num_sat 


def eval(formula,
         num_variables,
         variables,
         policy_network,
         device,
         strategy = 'greedy',
         batch_size = 1,
         permute_vars = False,
         permute_seed = None):

    policy_network.eval()
    policy_network.to(device)

    e_eval = Episode(formula,
                num_variables,
                variables,
                policy_network,
                device,
                strategy,
                batch_size,
                permute_vars,
                permute_seed,
                baseline = None,
                entropy_weight = 0)

    e_eval.initialize()
    assignment = e_eval.prediction()
    loss, mean_num_sat = e_eval.get_loss()

    return loss.detach(), mean_num_sat, assignment.detach()
    # ::assignment:: [batch_size, seq_len=num_variables]



def train(formula,
          num_variables,
          variables,
          num_episodes,
          accumulation_steps,
          policy_network,
          optimizer,
          device,
          strategy = 'sampled',
          batch_size = 1,
          permute_vars = False,
          permute_seed = None,
          baseline = None,
          entropy_weight = 0,
          clip_grad = None,
          verbose = 1,
          raytune = False,
          episode_log = False,
          episode_log_step = 1,
          optimizer_log = False,
          optimizer_log_step = 1,
          experiment_name = None):
    """ Train Enconder-Decoder policy following Policy Gradient Theorem
    
    PARAMETERS
    ----------

    episode_log : bool. Enable logging TensorBoard files (loss, num_sat, prob_0 and prob_1).
                    Default False.
    episode_log_step : int. If episode_log is True, log info every log_step steps. Default 1.

    logs_dir : str. Directory where tensorboard logs are saved. If None (default), logs are saved
                    as './outputs/logs/run' + time of the system.
    """

    # Initliaze parameters
    #def xavier_init_weights(m):
    #    if type(m) == nn.Linear:
    #        nn.init.xavier_uniform_(m.weight)
    #    if type(m) == nn.GRU or type(m) == nn.LSTM:
    #        for param in m._flat_weights_names:
    #            if "weight" in param:
    #                nn.init.xavier_uniform_(m._parameters[param])
    #policy_network.apply(xavier_init_weights)
    # TODO: check TrainableState
    # TODO: check initialize params
    # TODO: clean paths with os and opts.
    
    output_dir = 'outputs'
    log_dir = 'logs'
    exp_name = experiment_name
    if experiment_name is None:
        exp_name = 'exp_' + time.strftime("%Y%m%dT%H%M%S")
    run_id = time.strftime("%Y%m%dT%H%M%S")
    run_name = 'n' + str(num_variables)

    if episode_log:
        #log_dir = './outputs/' + experiment_name + '/runs/n' + str(num_variables) +'/'+str(r)
        #log_dir = './outputs/' + runname
        writer = SummaryWriter(log_dir = os.path.join(log_dir, exp_name, run_id + "_episode_log_" + run_name))
    
    if optimizer_log:
        writer_opt = SummaryWriter(log_dir = os.path.join(log_dir, exp_name, run_id + "_optimizer_log_" + run_name))

    policy_network.to(device)
    policy_network.train()
    optimizer.zero_grad()

    history_loss = []
    history_num_sat = []
    history_loss_val = []
    history_num_sat_val = []

    mean_loss = 0
    mean_num_sat = 0
    optim_step = 0  # counts the number of times optim.step() is applied

    for episode in range(1, num_episodes + 1):
        e_train = Episode(formula,
                        num_variables,
                        variables,
                        policy_network,
                        device,
                        strategy,
                        batch_size,
                        permute_vars,
                        permute_seed,
                        baseline,
                        entropy_weight)

        e_train.initialize()
        assignment = e_train.prediction()
        episode_loss, episode_num_sat = e_train.get_loss()

        """
        if verbose == 3:
            print(f"\nepisode: {episode}")
            print(f"logits: {episode_stats['logits']}")
            print(f"probs: {episode_stats['probs']}")
            print(f"sampled actions: {episode_stats['sampled actions']}")
            print(f"entropy: {episode_stats['entropy']}")
            print(f"weighted_entropy: {episode_stats['weighted_entropy']}")
            print(f"num_sat: {episode_stats['num_sat']}")
            print(f"baseline: {episode_stats['baseline']}")
            print(f"log_probs: {episode_stats['log_probs']}")
            print(f"loss: {episode_stats['loss']}")
        """

        # Normalize loss for gradient accumulation
        loss = episode_loss / accumulation_steps
        num_sat = episode_num_sat / accumulation_steps
        
        # Gradient accumulation
        loss.backward()

        # Accumulate mean loss and mean num sat
        mean_loss += loss.item()
        mean_num_sat += num_sat

        # Episode logging (logging every episode_log_step episodes)
        if episode_log and ((episode % episode_log_step) == 0):
            writer.add_scalar('loss', episode_loss.item(), episode, new_style=True)
            writer.add_scalar('num_sat', episode_num_sat, episode, new_style=True)

            for (prob_0, prob_1) in e_train.buffer.action_softmax[0]:
                writer.add_scalar('prob_0', prob_0.item(), episode, new_style=True)
                writer.add_scalar('prob_1', prob_1.item(), episode, new_style=True)

        # Perform optimization step after accumulating gradients
        if (episode % accumulation_steps) == 0:
            
            # Optimizer step
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(policy_network.parameters(), clip_grad) 
            optimizer.step()
            optimizer.zero_grad()
            optim_step += 1

            # Validation
            policy_network.eval()
            with torch.no_grad():
                val_loss, val_num_sat, val_assignment = eval(formula,
                                                            num_variables,
                                                            variables,
                                                            policy_network,
                                                            device,
                                                            strategy = 'greedy',
                                                            batch_size = 1,
                                                            permute_vars = permute_vars,
                                                            permute_seed = permute_seed)


            policy_network.train()

            # Optimizer logging (logging every accumulation_step * optimizer_log_step episodes, e.x., every optimizer step when optimizer_log_step=1)
            if optimizer_log and ((optim_step % optimizer_log_step) == 0):
                writer_opt.add_scalar('mean_loss', mean_loss, episode, new_style=True)
                writer_opt.add_scalar('mean_num_sat', mean_num_sat, episode, new_style=True)
                writer_opt.add_scalar('val_loss', val_loss, episode, new_style=True)
                writer_opt.add_scalar('val_num_sat', val_num_sat, episode, new_style=True)

            # Trackers (every accumulation_step episodes)
            #history_loss.append(mean_loss)
            #history_num_sat.append(mean_num_sat)
            #history_loss_val.append(val_loss)
            #history_num_sat_val.append(val_num_sat)
            
        
            if verbose == 2 or verbose == 3 or ((verbose == 1) and (episode == num_episodes)):
                print(f'\nGreedy actions (train): {assignment}')
                print(f'\nGreedy actions (val): {val_assignment}')

                print('Optim step {}, Episode [{}/{}], Mean loss {:.4f},  Mean num sat {:.4f}, Val loss {:.4f}, Val num sat {:.4f}' 
                        .format(optim_step, episode, num_episodes, mean_loss, mean_num_sat, val_loss, val_num_sat))
            
        
            if raytune:
                with tune.checkpoint_dir(episode) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((policy_network.state_dict(), optimizer.state_dict()), path)

                tune.report(optim_step=optim_step, episode=episode, loss=mean_loss, num_sat=mean_num_sat, num_sat_val=val_num_sat)

            # Reset 
            mean_loss = 0
            mean_num_sat = 0
    
    if episode_log:
        writer.close()
    if optimizer_log:
        writer_opt.close()