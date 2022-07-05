import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from src.utils import assignment_verifier
from src.utils import sampling_assignment

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from ray import tune
import os

def forward_episode(formula,
                    num_variables,
                    variables,
                    policy_network,
                    device,
                    baseline = None,
                    entropy_weight = 0):
    # Encoder
    enc_output = None
    if policy_network.encoder is not None:
        enc_output = policy_network.encoder(formula, num_variables, variables)

    # Initialize Decoder Variables 
    var = policy_network.init_dec_var(enc_output, formula, num_variables, variables)
    # ::var:: [batch_size, seq_len, feature_size]

    batch_size = var.shape[0]

    # Initialize action_prev at time t=0 with token 2.
    #   Token 0 is for assignment 0, token 1 for assignment 1
    action_prev = torch.tensor([2] * batch_size, dtype=torch.long).reshape(-1,1,1).to(device)
    # ::action_prev:: [batch_size, seq_len=1, feature_size=1]

    # Initialize Decoder Context
    context = policy_network.init_dec_context(enc_output, formula, num_variables, variables, batch_size).to(device)
    # ::context:: [batch_size, feature_size]

    # Initialize Decoder state
    state = policy_network.init_dec_state(enc_output, batch_size)
    if state is not None: 
        state = state.to(device)
    # ::state:: [num_layers, batch_size, hidden_size]
    
    actions_logits = []  # for debugging verbose 3
    actions_softmax = []  # for debugging verbose 3
    actions = []
    action_log_probs = []
    entropies = []



    for t in range(num_variables):
        var_t = var[:,t:t+1,:].to(device)

        # Action logits
        action_logits, state = policy_network.decoder((var_t, action_prev, context), state)
        # ::action_logits:: [batch_size=1, seq_len=1, feature_size=2]

        # Prob distribution over actions
        #action_softmax = F.softmax(action_logits, dim = -1)  
        action_dist = distributions.Categorical(logits= action_logits)
        
        # Sample a rondom action 
        action = action_dist.sample()
        
        # Log-prob of the action
        action_log_prob = action_dist.log_prob(action)

        # Computing Entropy
        action_dist_entropy = action_dist.entropy()
        
        # Take the choosen action
        #-------

        # Store actions and action_log_prob
        actions_logits.append(list(np.around(action_logits.detach().cpu().numpy().flatten(), 2)))  # for debugging verbose 3
        actions_softmax.append(list(np.around(F.softmax(action_logits.detach(), -1).numpy().flatten(), 2)))  # for debugging verbose 3
        
        actions.append(action.item())
        #TODO: entropies detach?
        action_log_probs.append(action_log_prob)
        entropies.append(action_dist_entropy)

        action_prev = action.unsqueeze(dim=-1)
        #::action_prev:: [batch_size, seq_len=1, feature_size=1]
    
    policy_network.eval()
    with torch.no_grad():
        # Compute num of sat clauses
        is_sat, num_sat, _ = assignment_verifier(formula, actions)
        num_sat = torch.tensor(num_sat, dtype=float).detach()

        #TODO: Test baseline
        #Compute baseline
        baseline_val = torch.tensor(0, dtype=float).detach()
        if baseline is not None:
            #baseline_val = baseline(formula, torch.stack(action_logits_list)).detach()
            baseline_val = baseline(formula, num_variables, variables, policy_network, device).detach()

    policy_network.train()
    
    #Get loss
    total_action_log_probs = torch.cat(action_log_probs).sum()  # TODO: torch.cat() or torch.tensor()
    total_entropy = torch.tensor(entropies).sum()
    loss = - ((num_sat - baseline_val) * total_action_log_probs) - (entropy_weight * total_entropy)
    
    stats = {'loss': loss.item(),
            'num_sat': num_sat.item(),
            'logits': actions_logits,
            'probs': actions_softmax,
            'sampled actions': actions,
            'entropy': total_entropy.item(),
            'weighted_entropy': (entropy_weight * total_entropy).item(),
            'baseline': baseline_val.item(),
            'log_probs': total_action_log_probs.item()}
    
    return loss, stats


def train(formula,
          num_variables,
          variables,
          num_episodes,
          accumulation_steps,
          policy_network,
          optimizer,
          device,
          baseline = None,
          entropy_weight = 0,
          clip_grad = None,
          verbose = 1,
          raytune = False,
          episode_logs = None,
          logs_steps = 1):
    """ Train Enconder-Decoder policy following Policy Gradient Theorem"""

    # Initliaze parameters
    #def xavier_init_weights(m):
    #    if type(m) == nn.Linear:
    #        nn.init.xavier_uniform_(m.weight)
    #    if type(m) == nn.GRU or type(m) == nn.LSTM:
    #        for param in m._flat_weights_names:
    #            if "weight" in param:
    #                nn.init.xavier_uniform_(m._parameters[param])
    #policy_network.apply(xavier_init_weights)
    #TODO: check TrainableState
    #TODO: check initialize params

    if episode_logs is not None:
    #log_dir = './outputs/' + experiment_name + '/runs/n' + str(num_variables) +'/'+str(r)
        log_dir = './outputs/' + 'exp_1'
        writer = SummaryWriter(log_dir = log_dir)

  
    policy_network.to(device)
    policy_network.train()
    optimizer.zero_grad()

    history_loss = []
    history_num_sat = []
    hitosry_num_sat_val = []

    mean_loss = 0
    mean_num_sat = 0
    
    optim_step = 0  # counts the number of times optim.step() is applied

    for episode in range(1, num_episodes + 1):
        
        episode_loss, episode_stats = forward_episode(formula,
                                                    num_variables,
                                                    variables,
                                                    policy_network,
                                                    device,
                                                    baseline,
                                                    entropy_weight)
        
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
            
        # Gradient accumulation
        loss = episode_loss / accumulation_steps
        loss.backward()

        # Accumulate mean loss and mean num sat
        mean_loss += loss.item()
        mean_num_sat += (episode_stats['num_sat'] / accumulation_steps)

        if (episode_logs == 'loss_and_sat') and ((episode % logs_steps) == 0):
            writer.add_scalar('loss', episode_stats['loss'], episode, new_style=True)
            writer.add_scalar('num_sat', episode_stats['num_sat'], episode, new_style=True)

        if (episode_logs == 'probs') and ((episode % logs_steps) == 0):
            for (prob_0, prob_1) in episode_stats['probs']:
                #print(prob_0, prob_1)
                writer.add_scalar('prob_0', prob_0, episode, new_style=True)
                writer.add_scalar('prob_1', prob_1, episode, new_style=True)





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
                strategy='greedy'
                assignment = sampling_assignment(formula, num_variables, variables,
                                                policy_network, device, strategy)
                _, num_sat_val, _ = assignment_verifier(formula, assignment)
            policy_network.train()

            # Trackers (every accumulation_step episodes)
            history_loss.append(mean_loss)
            history_num_sat.append(mean_num_sat)
            hitosry_num_sat_val.append(num_sat_val)
            
        
            if verbose == 2 or verbose == 3 or ((verbose == 1) and (episode == num_episodes)):
                print(f'\nGreedy actions: {assignment}')
                print('Optim step {}, Episode [{}/{}], Mean loss {:.4f},  Mean num sat {:.4f}, Val num sat {:.4f}' 
                        .format(optim_step, episode, num_episodes, mean_loss, mean_num_sat, num_sat_val))
            
        
            if raytune:
                with tune.checkpoint_dir(episode) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((policy_network.state_dict(), optimizer.state_dict()), path)

                tune.report(optim_step=optim_step, episode=episode, loss=mean_loss, num_sat=mean_num_sat, num_sat_val=num_sat_val)

            # Reset 
            mean_loss = 0
            mean_num_sat = 0
    
    if episode_logs is not None:
        writer.close()
    
    return history_loss, history_num_sat, hitosry_num_sat_val