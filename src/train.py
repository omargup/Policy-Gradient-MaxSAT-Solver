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
          variables,
          num_episodes,
          policy_network,
          optimizer,
          device,
          baseline = None,
          input_seq = None,
          context = None,
          entropy_weight = 0,
          clip_val = None,
          verbose = 1):
    
    """ Train Enconder-Decoder policy"""

    # Initliaze parameters
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU or type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    policy_network.apply(xavier_init_weights)

    policy_network.to(device)
    policy_network.train()
    optimizer.zero_grad()

    history_loss = []
    history_num_sat = []

    mean_loss = 0
    mean_num_sat = 0

    for episode in range(1, num_episodes + 1):
        # Encoder
        enc_output = None
        if policy_network.encoder is not None:
            enc_output = policy_network.encoder(formula, num_variables, variables)

        # Initialize Decoder Variables 
        var = policy_network.init_dec_var(enc_output, formula, num_variables, variables, device)
        # ::var:: [batch_size, seq_len, feature_size]

        batch_size = var.shape[0]

        # Initialize action_prev at time t=0 with token 2.
        #   Token 0 is for assignment 0, token 1 for assignment 1
        action_prev = torch.tensor([2] * batch_size, dtype=torch.long, device=device).reshape(-1,1,1)
        # ::action_prev:: [batch_size, seq_len=1, feature_size=1]

        # Initialize Decoder Context
        context = policy_network.init_dec_context(enc_output, formula, num_variables, variables, batch_size, device)
        # ::context:: [batch_size, feature_size]

        # Initialize Decoder state
        state = policy_network.init_dec_state(enc_output)
        # ::state:: [num_layers, batch_size=1, hidden_size]
        
        action_log_probs = []
        actions = []
        action_logits_list = []  # useful for baseline
        actions_logits = []  # for debugging
        entropy_list = []
        

        for t in range(num_variables):
            #TODO: send to device here.
   
            # Action logits
            action_logits, state= policy_network.decoder((var[:,t:t+1,:], action_prev, context), state)

            # Prob distribution over actions
            #action_softmax = F.softmax(action_logits, dim = -1)  
            action_dist = distributions.Categorical(logits= action_logits)
            
            # Sample a rondom action 
            action = action_dist.sample()
            
            # Log-prob of the action
            action_log_prob = action_dist.log_prob(action)

            # Computing Entropy
            entropy = action_dist.entropy()
            
            # Take the choosen action
            #-------

            # Store actions and action_log_prob
            action_log_probs.append(action_log_prob)
            actions.append(action.item())
            action_logits_list.append(action_logits)
            actions_logits.append(list(np.around(action_logits.detach().numpy().flatten(), 2)))  # for debugging
            #actions_logits.append(list(np.around(F.softmax(action_logits.detach(), -1).numpy().flatten(), 2)))  # for debugging
            entropy_list.append(entropy)

            action_prev = action
    
        with torch.no_grad():
            # Compute num of sat clauses
            is_sat, num_sat, _ = assignment_verifier(formula, actions)
            num_sat = torch.tensor(num_sat, dtype=float).detach()

            #TODO: Check baseline, encoder-decoder
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