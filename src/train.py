#from generator import UniformCNFGenerator
from src.utils import assignment_verifier
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
#import torch.optim as optim

from src.architectures import BasicRNN


def train(accumulation_steps,
          formula,
          num_variables,
          num_episodes,
          policy_network,
          optimizer,
          baseline = None,
          entropy_weight = 0,
          clip_logits = None,
          clip_val = None,
          verbose = 1):
    
    history_loss = []
    history_num_sat = []

    mean_loss = 0
    mean_num_sat = 0

    policy_network.train()
    optimizer.zero_grad()
    for episode in range(1, num_episodes + 1):

        #Create the input: a sequence of integers from 0 to n-1,
        # representing X = [[x0, x1, ..., xn]]
        X = torch.tensor([[i for i in range(num_variables)]])
        #X: [batch_size=1, seq_len]
        
        action_log_probs = []
        actions = []
        action_logits_list = []  # useful for baseline
        actions_logits = []  # for debugging
        entropy_list = []

        X = X.permute(1, 0).unsqueeze(-1)
        #X: [seq_len, batch_size=1]

        #action_prev is the action taken at time t-1. 
        # At time t=0, a_prev=2 which means no action
        # has been taken.
        action_prev = torch.tensor(2).reshape(1,1) 
        #action_prev:[batch_size=1, seq_len=1]

        state = policy_network.init_state_basicrnn()
        #state: [num_layers, batch_size, hidden_size]

        for t, x in enumerate(X):
            #x: [seq_len=1, batch_size=1]
            
            x = x.permute(1, 0)
            #x: [batch_size=1, seq_len=1]

            input_t = (x, action_prev)
            
            #Action logits
            action_logits, state = policy_network((input_t), state)

            #TODO ERROR MESSAGE
            if clip_logits is not None:
                action_logits = clip_logits * F.tanh(action_logits)

            #Prob distribution over actions
            #action_softmax = F.softmax(action_logits, dim = -1)
            
            #Probability distribution over actions    
            action_dist = distributions.Categorical(logits= action_logits)
            
            #Sample a rondom action 
            action = action_dist.sample()
            
            #Log-prob of the action
            action_log_prob = action_dist.log_prob(action)

            # Computing Entropy
            entropy = action_dist.entropy()
            
            #Take the choosen action
            #-------

            #Store actions and action_log_prob
            action_log_probs.append(action_log_prob)
            actions.append(action.item())
            action_logits_list.append(action_logits)
            actions_logits.append(list(np.around(action_logits.detach().numpy().flatten(), 2)))  # for debugging
            #actions_logits.append(list(np.around(F.softmax(action_logits.detach(), -1).numpy().flatten(), 2)))  # for debugging
            entropy_list.append(entropy)

            action_prev = action
    
        #Compute num of sat clauses
        is_sat, num_sat, _ = assignment_verifier(formula, actions)
        num_sat = torch.tensor(num_sat, dtype=float).detach()

        #Compute baseline
        baseline_val = torch.tensor(0, dtype=float)
        if baseline is not None:
            #baseline_val = baseline(formula, torch.stack(action_logits_list)).detach()
            baseline_val = baseline(formula, policy_network, num_variables).detach()
    

        #Get loss
        action_log_probs = torch.cat(action_log_probs)
        entropy_list = torch.tensor(entropy_list)
        loss = - ((num_sat - baseline_val) * action_log_probs.sum() + entropy_weight * entropy_list.sum())

        #Gradient accumulation
        loss = loss / accumulation_steps
        num_sat = num_sat / accumulation_steps
        loss.backward()

        mean_loss += loss.item()
        mean_num_sat += num_sat
        
        if verbose == 3:
            print(actions_logits)

        if (episode % accumulation_steps) == 0:
            if clip_val is not None:
                nn.utils.clip_grad_norm_(policy_network.parameters(), clip_val) 
            optimizer.step()
            optimizer.zero_grad()
        
            #Trackers
            history_loss.append(mean_loss)
            history_num_sat.append(mean_num_sat)
        
            if verbose == 2 or verbose == 3:
                print('Episode [{}/{}], Mean Loss {:.4f},  Mean num sat {:.4f}' 
                        .format(episode, num_episodes, mean_loss, mean_num_sat))
            
            if verbose == 1 and episode== num_episodes:
                print('Episode [{}/{}], Mean Loss {:.4f},  Mean num sat {:.4f}' 
                        .format(episode, num_episodes, mean_loss, mean_num_sat))
        
            mean_loss = 0
            mean_num_sat = 0
    
    
        
    return history_loss, history_num_sat