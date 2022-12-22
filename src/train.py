from random import betavariate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from src.initializers.state_initializer import TrainableState
import src.utils as utils
from src.utils import sampling_assignment

import numpy as np
from ray import tune

from ray.air import session
from ray.air.checkpoint import Checkpoint

import os
import time
from tqdm import tqdm



# TODO: save only sum, not each value.
class Buffer():
    """
    Tracks episode's relevant information.
    """
    def __init__(self, batch_size, num_variables, dec_output_size) -> None:
        # Episode Buffer
        self.action_logits = torch.empty(size=(batch_size, num_variables, dec_output_size))
        # ::buffer_action_logits:: [batch_size, seq_len=num_variables, feature_size=1or2]
        self.action_probs = torch.empty(size=(batch_size, num_variables, dec_output_size))
        # ::buffer_action_probs:: [batch_size, seq_len=num_variables, feature_size=1or2]
        self.action = torch.empty(size=(batch_size, num_variables), dtype = torch.int64)
        # ::buffer_action:: [batch_size, seq_len=num_variables]
        self.action_log_prob = torch.empty(size=(batch_size, num_variables))
        # ::buffer_action_log_prob:: [batch_size, seq_len=num_variables]
        self.entropy = torch.empty(size=(batch_size, num_variables))
        # ::buffer_entropy:: [batch_size, seq_len=num_variables]
    
    def update(self, idx, t, action_logits, action_probs, action, action_log_prob, entropy):
        assert action.dtype == torch.int64, f'action in update Buffer. dtype: {action.dtype}, shape: {action.shape}.'
        self.action_logits[idx, t] = action_logits.squeeze(1)
        self.action_probs[idx, t] = action_probs.squeeze(1)
        self.action[idx, t] = action.view(-1)
        self.action_log_prob[idx, t] = action_log_prob.view(-1)
        self.entropy[idx, t] = entropy.view(-1)


def vars_permutation(num_variables,
                    device,
                    batch_size = 1,
                    permute_vars=False,
                    permute_seed=None):  # e.g.: 2147483647
    """
    Returns a permutation of the variables' indices.
    """
    if permute_vars:
        if permute_seed is not None:
            gen = torch.Generator(device=device)
            permutation = torch.cat([torch.randperm(num_variables, generator=gen.manual_seed(permute_seed)).unsqueeze(0) for _ in range(batch_size)], dim=0).permute(1,0)
        else:
            permutation = torch.cat([torch.randperm(num_variables).unsqueeze(0) for _ in range(batch_size)], dim=0).permute(1,0)
    else:
        permutation = torch.cat([torch.tensor([i for i in range(num_variables)]).unsqueeze(0) for _ in range(batch_size)], dim=0).permute(1,0)
        # ::permutation:: [num_variables, batch_size]; e.g.: [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]

    return permutation


def run_episode(num_variables,
                policy_network,
                device,
                dec_init_state,
                dec_vars,
                dec_context,
                init_action,
                strategy = 'sampled',
                batch_size = 1,
                permute_vars = False,
                permute_seed = None):  # e.g.: 2147483647          
    """
    Runs an episode and returns an updated buffer.
    """
    dec_output_size = policy_network.decoder.dense_out.bias.shape[0]
    assert (dec_output_size == 1) or (dec_output_size == 2), f"In run_episode. Decoder's output shape[-1]: {dec_output_size}"
    buffer = Buffer(batch_size, num_variables, dec_output_size)
    batch_idx = [i for i in range(batch_size)]

    permutation = vars_permutation(num_variables,
                                   device,
                                   batch_size,
                                   permute_vars,
                                   permute_seed)
    # ::permutation:: [num_variables, batch_size]; # e.g.: [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    
    # Expanding dec_init_state 
    state = None
    if dec_init_state is not None:
        if type(state) == tuple:  # LSTM
            state = (dec_init_state[0].expand(dec_init_state[0].shape[0], batch_size, dec_init_state[0].shape[2]), \
                dec_init_state[1].expand(dec_init_state[1].shape[0], batch_size, dec_init_state[1].shape[2]))
            # ::state h:: [num_layers, batch_size, hidden_size]
            # ::state c:: [num_layers, batch_size, hidden_size]
        else:  # GRU
            state = dec_init_state.expand(dec_init_state.shape[0], batch_size, dec_init_state.shape[2])
            # ::state:: [num_layers, batch_size, hidden_size]

    # Expanding init_action
    #action_prev = torch.cat([init_action] * batch_size)
    action_prev = init_action.expand(batch_size, init_action.shape[1], init_action.shape[2])
    # ::action_prev:: [batch_size, seq_len=1, feature_size=1]
    assert action_prev.dtype == torch.int64, f'Expanding action_prev in run_episode. dtype: {action_prev.dtype}, shape: {action_prev.shape}.'

    # Expanding dec_vars
    #dec_vars = torch.cat([dec_vars] * batch_size)
    dec_vars = dec_vars.expand(batch_size, dec_vars.shape[1], dec_vars.shape[2])
    # ::dec_vars:: [batch_size, seq_len=num_variables, feature_size]

    # Expanding dec_context
    #dec_context = torch.cat([dec_context] * batch_size)
    dec_context = dec_context.expand(batch_size, -1)
    # ::dec_context:: [batch_size, feature_size]


    for var_idx in permutation:
        var_t = dec_vars[batch_idx, var_idx].unsqueeze(1).to(device)
        assert var_t.shape == (batch_size, 1, dec_vars.shape[-1])
        # ::var_t:: [batch_size, seq_len=1, feature_size]

        # Action logits
        action_logits, state = policy_network.decoder((var_t, action_prev, dec_context), state)
        # ::action_logits:: [batch_size, seq_len=1, output_size=(1 or 2)]
        assert ((action_logits.shape[-1] == 1) or (action_logits.shape[-1] == 2)), f'action_logits in run_episode. shape: {action_logits.shape}.'

        # Prob distribution over actions
        if action_logits.shape[-1] == 1:
            action_probs = torch.sigmoid(action_logits)
            action_dist = distributions.Bernoulli(probs=action_probs)
        else:  # action_logits.shape[-1] == 2
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = distributions.Categorical(probs=action_probs)
        
        # Action selection
        if strategy == 'greedy':
            if action_logits.shape[-1] == 1:
                action = torch.round(action_probs)
            else:  # action_logits.shape[-1] == 2
                action = torch.argmax(action_probs, dim=-1)
        elif strategy == 'sampled':
            action = action_dist.sample()
        else:
            raise TypeError("{} is not a valid strategy, try with 'greedy' or 'sampled'.".format(strategy))
        assert action.shape == torch.Size([batch_size, 1, 1])
        # ::action:: [batch_size, seq_len=1, feature_size=1]

        # Log-prob of the action
        action_log_prob = action_dist.log_prob(action)

        # Computing Entropy
        entropy = action_dist.entropy()
        
        # Take the choosen action
        #-------

        # Update buffer
        buffer.update(batch_idx, var_idx, action_logits, action_probs, action.to(dtype=torch.int64), action_log_prob, entropy)
        
        #actions_logits.append(list(np.around(action_logits.detach().cpu().numpy().flatten(), 2)))
        #actions_softmax.append(list(np.around(F.softmax(action_logits.detach(), -1).numpy().flatten(), 2)))
        #actions.append(action.item())
        #action_log_probs.append(action_log_prob)
        #entropies.append(action_dist_entropy)

        action_prev = action.to(dtype=torch.int64)
        #::action_prev:: [batch_size, seq_len=1, feature_size=1]
        assert action.shape == torch.Size([batch_size, 1, 1])
    
    # return buffer.action  # self.buffer.action.detach().cpu().numpy()
    return buffer
    

def train(formula,
          num_variables,
          variables,
          policy_network,
          optimizer,
          device,
          strategy='sampled',
          batch_size=1,
          permute_vars=False,
          permute_seed=None,
          baseline=None,
          entropy_weight=0,
          clip_grad=None,
          raytune = False,
          num_episodes=5000,
          accumulation_episodes=1,
          log_episodes=100,
          eval_episodes=100,
          eval_strategies=[0, 10], # 0 for greedy, i < 0 takes i samples and returns the best one.
          writer = None,  # Tensorboard writer
          extra_logging = False,
          run_name = None,
          progress_bar = False):
    """ Train Enconder-Decoder policy following Policy Gradient Theorem
    
    PARAMETERS
    ----------
    log_episodes : Enable logging TensorBoard files (loss, num_sat, prob_0 and prob_1).
                    Default False.
    log_episodes : int. Log info every `log_episodes` episodes. Default: 100.

    eval_episodes : int. Run evaluation every `eval_episodes` episodes. Default: 100.

    logs_dir : str. Directory where tensorboard logs are saved. If None (default), logs are saved
                    as './outputs/logs/run' + time of the system.
    """
    print(f"\nStart training for run-id {run_name}")

    # Put model in train mode
    policy_network.to(device)
    policy_network.train()
    optimizer.zero_grad()

    # Initializations (for training)
    # Initialize Encoder
    enc_output = None
    if policy_network.encoder is not None:
        enc_output = policy_network.encoder(formula, num_variables, variables)

    # Initialize Decoder Variables 
    dec_vars = policy_network.dec_var_initializer(enc_output, formula, num_variables, variables)
    # ::dec_vars:: [batch_size=1, seq_len=num_variables, feature_size]

    # Initialize Decoder Context
    dec_context = policy_network.dec_context_initializer(enc_output, formula, num_variables, variables).to(device)
    # ::dec_context:: [batch_size=1, feature_size]

    # Initialize action_prev at time t=0 with token 2.
    #   Token 0 is for assignment 0, token 1 for assignment 1
    init_action = torch.tensor([2], dtype=torch.long).reshape(-1,1,1).to(device)
    # ::init_action:: [batch_size=1, seq_len=1, feature_size=1]

    # Initialize Decoder state
    dec_init_state = policy_network.dec_state_initializer(enc_output)
    if dec_init_state is not None: 
        dec_init_state = dec_init_state.to(device)
    # ::dec_init_state:: [num_layers, batch_size=1, hidden_size]


    for episode in tqdm(range(1, num_episodes + 1), disable=progress_bar, ascii=True):
        
        buffer = run_episode(num_variables,
                             policy_network,
                             device,
                             dec_init_state,
                             dec_vars,
                             dec_context,
                             init_action,
                             strategy,
                             batch_size,
                             permute_vars,
                             permute_seed)
        
        policy_network.eval()
        with torch.no_grad():
            # Compute num of sat clauses
            #num_sat = utils.num_sat_clauses_tensor(formula, buffer.action.detach().cpu().numpy()).detach()
            num_sat = utils.num_sat_clauses_tensor(formula, buffer.action.detach()).detach()
            # ::num_sat:: [batch_size]

            # Compute baseline
            baseline_val = torch.tensor(0, dtype=float).detach()
            #if baseline is not None:
            #    baseline_val = baseline(formula, num_variables, variables, policy_network, device, permute_vars, permute_seed).detach()

        policy_network.train()
        
        # Get loss (mean over the batch)
        mean_action_log_prob =  buffer.action_log_prob.sum(dim=-1).mean()
        mean_entropy = buffer.entropy.sum(-1).mean() #.detach()
        pg_loss = - ((num_sat.mean() - baseline_val) * mean_action_log_prob) 
        pg_loss_with_ent = pg_loss - (entropy_weight * mean_entropy)
        
        # Normalize loss for gradient accumulation
        loss = pg_loss_with_ent / accumulation_episodes
        #num_sat = num_sat / accumulation_episodes
        
        # Gradient accumulation
        loss.backward()

        # Perform optimization step after accumulating gradients
        if (episode % accumulation_episodes) == 0:
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(policy_network.parameters(), clip_grad) 
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        if (episode % log_episodes) == 0:

            num_sat_mean = num_sat.mean().item()

            # Log values to screen
            print(f'\nEpisode: {episode}, num_sat: {num_sat_mean}')
            print('pg_loss: ({} - {}) * {} = {}'.format(num_sat_mean,
                                                        baseline_val.item(),
                                                        mean_action_log_prob.item(),
                                                        pg_loss.item()))
            print('pg_loss + entropy: {} + ({} * {}) = {}'.format(pg_loss.item(),
                                                                entropy_weight,
                                                                mean_entropy.item(),
                                                                pg_loss_with_ent.item()))
            
            if writer is not None:
                writer.add_scalar('num_sat', num_sat_mean, episode, new_style=True)
                writer.add_scalar('pg_loss', pg_loss.item(), episode, new_style=True)
                writer.add_scalar('pg_loss_with_ent', pg_loss_with_ent.item(), episode, new_style=True)
                writer.add_scalar('log_prob', mean_action_log_prob.item(), episode, new_style=True)
                writer.add_scalar('baseline', baseline_val.item(), episode, new_style=True)
                writer.add_scalar('entropy/entropy', mean_entropy.item(), episode, new_style=True)
                writer.add_scalar('entropy/w*entropy', (entropy_weight * mean_entropy).item(), episode, new_style=True)

                if extra_logging:
                    writer.add_histogram('histogram/action_logits', buffer.action_logits, episode)
                    writer.add_histogram('histogram/action_probs', buffer.action_probs, episode)
                    
                    if type(policy_network.dec_state_initializer) == TrainableState:
                        writer.add_histogram('params/init_state', policy_network.dec_state_initializer.h, episode)


        # Validation
        if (episode % eval_episodes) == 0:
            print(f'\n-------------------------------------------------')
            print(f'Evaluation in episode: {episode}. Num of sat clauses:')
            policy_network.eval()
            with torch.no_grad():
                
                for strat in eval_strategies:
                    #TODO: Do not allow negative values for strat
                    buffer = run_episode(num_variables = num_variables,
                                        policy_network = policy_network,
                                        device = device,
                                        dec_init_state = dec_init_state,
                                        dec_vars = dec_vars,
                                        dec_context = dec_context,
                                        init_action = init_action,
                                        strategy = 'greedy' if strat == 0 else 'sampled',
                                        batch_size = 1 if strat == 0 else strat,
                                        permute_vars = permute_vars,
                                        permute_seed = permute_seed)
            
                    # Compute num of sat clauses
                    num_sat = utils.num_sat_clauses_tensor(formula, buffer.action.detach()).detach()
                    # ::num_sat:: [batch_size]

                    # Log values to screen
                    if strat == 0:
                        number_of_sat = num_sat.item()
                        print(f'\tGreedy: {number_of_sat}.')
                    else:
                        number_of_sat = num_sat.max().item()
                        print(f'\tBest of {strat} samples: {number_of_sat}.')
                    
                    if writer is not None:
                        writer.add_scalar(f"eval/{'greedy' if strat == 0 else 'sampled'}{'' if strat == 0 else '-'+str(strat)}",
                                          number_of_sat, episode, new_style=True)
                
            
                    #  for v in num_variables:
        #                 writer.add_scalars('buffer/actions', {'xsinx':i*np.sin(i/r),
        #                                                         'xcosx':i*np.cos(i/r),
        #                                                         'tanx': np.tan(i/r)}, i)
                        
        #             writer.add_scalars('buffer/action_logits', {'xsinx':i*np.sin(i/r),
        #                                                         'xcosx':i*np.cos(i/r),
        #                                                         'tanx': np.tan(i/r)}, i)
        # self.action_logits = torch.empty(size=(batch_size, num_variables, dec_output_size))
        # # ::buffer_action_logits:: [batch_size, seq_len=num_variables, feature_size=1or2]
        # self.action_probs = torch.empty(size=(batch_size, num_variables, dec_output_size))
        # # ::buffer_action_probs:: [batch_size, seq_len=num_variables, feature_size=1or2]
        # self.action = torch.empty(size=(batch_size, num_variables), dtype = torch.int64)
                            
            print(f'\n-------------------------------------------------')
            policy_network.train()





        # for prob in buffer.action_probs[0]:
        #     writer.add_scalar('prob', prob.item(), episode, new_style=True)

        # if dec_output_size == 1:

        # else:
        #     writer.add_scalars('probs', {'var'+str(i): probs[i] for i in range(5)}, episode, new_style=True)

        # for i in range(num_variables):
        #     write.add_scalar('probs/var'+str(i), probs[i], episode, new_style=True)
        #     write.add_scalar('probs/var'+str(i), probs[i], episode, new_style=True)


            # if raytune:
            #     #with tune.checkpoint_dir(episode) as checkpoint_dir:
            #     #    path = os.path.join(checkpoint_dir, "checkpoint")
            #     #    torch.save((policy_network.state_dict(), optimizer.state_dict()), path)

            #     ray_checkpoints_path = os.path.join('ray_results/checkpoints')
            #     if not os.path.exists(ray_checkpoints_path):
            #         os.makedirs(ray_checkpoints_path)

            #     #path = "/Users/omargutierrez/Documents/Code/learning_sat_solvers/my_model"
            #     #os.makedirs("my_model", exist_ok=True)
            #     torch.save((policy_network.state_dict(), optimizer.state_dict()), 
            #                 os.path.join(ray_checkpoints_path, "checkpoint.pt"))
            #     checkpoint = Checkpoint.from_directory(ray_checkpoints_path)
                

            #     session.report({'optim_step':optim_step, 
            #                     'episode':episode,
            #                     'loss':mean_loss,
            #                     'num_sat':mean_num_sat,
            #                     'num_sat_val':val_num_sat},
            #                     checkpoint=checkpoint)

    
 
