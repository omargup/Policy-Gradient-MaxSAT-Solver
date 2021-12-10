from generator import UniformCNFGenerator
import utils
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
#import torch.optim as optim

from architectures import BasicRNN


def train(accumulation_steps,
          formula,
          num_variables,
          num_episodes,
          policy_network,
          optimizer,
          baseline=None,
          clip_val=None):
    
    history_loss = []
    history_num_sat = []

    mean_loss = 0
    mean_num_sat = 0

    policy_network.train()
    optimizer.zero_grad()
    for episode in range(num_episodes):

        X = torch.tensor([[i for i in range(num_variables)]])
        
        action_log_probs = []
        actions = []
        action_logits_list = [] #useful for baseline
        actions_logits = [] #for debugging***********************

        #1. X = [[x0, x1, ..., xn]]
        #X: [batch_size=1, seq_len]
        X = X.permute(1, 0).unsqueeze(-1)
        #X: [seq_len, batch_size=1]

        a_prev = torch.tensor(2).reshape(1,1) #2 means no a_prev
        #a_prev:[batch_size=1, seq_len=1]

        state = policy_network.init_state_basicrnn()
        #state: [num_layers, batch_size, hidden_size]

        for t, x in enumerate(X):
            #x: [seq_len=1, batch_size=1]
            x = x.permute(1, 0)
            #x: [batch_size=1, seq_len=1]

            input_t = (x, a_prev)
            #2. Action logits
            action_logits, state = policy_network((input_t), state)

            #3. Prob distribution over actions
            action_softmax = F.softmax(action_logits, dim = -1)
            #4. Sample an action        
            action_dist = distributions.Categorical(action_softmax)
            action = action_dist.sample()
            #5. Log-prob of the action
            action_log_prob = action_dist.log_prob(action)
            #6. Take the choosen action
            #-------
            #7 Store actions and action_log_prob
            action_log_probs.append(action_log_prob)
            actions.append(action.item())
            action_logits_list.append(action_logits)
            actions_logits.append(list(np.around(action_logits.detach().numpy().flatten(), 2))) #for debugging****************

            a_prev = action
    
        #Compute num of sat clauses
        is_sat, num_sat, _ = utils.assignment_verifier(formula, actions)
        num_sat = torch.tensor(num_sat, dtype=float).detach()

        #Compute baseline
        baseline_val = torch.tensor(0, dtype=float)
        if baseline is not None:
            baseline_val = baseline(formula, torch.stack(action_logits_list)).detach()

        #Get loss
        action_log_probs = torch.cat(action_log_probs)
        loss = - ((num_sat - baseline_val) * action_log_probs.sum())

        #Gradient accumulation
        loss = loss / accumulation_steps
        num_sat = num_sat / accumulation_steps
        loss.backward()

        mean_loss += loss.item()
        mean_num_sat += num_sat

        if (episode + 1) % accumulation_steps == 0:
            if clip_val is not None:
                nn.utils.clip_grad_norm_(policy_network.parameters(), clip_val) 
            optimizer.step()
            optimizer.zero_grad()
        
            #Trackers
            history_loss.append(mean_loss)
            history_num_sat.append(mean_num_sat)
        
            print('Episode [{}/{}], Loss {:.4f}, H {:.4f}, Actions {}, Actions logits {}' 
                    .format(episode+1, num_episodes, mean_loss, mean_num_sat, actions, actions_logits))
        
            mean_loss = 0
            mean_num_sat = 0
        
    return history_loss, history_num_sat



'''
#Create a sat generator
sat_gen = UniformCNFGenerator(min_n = 5,
                              max_n = 5,
                              min_k = 3,
                              max_k = 3,
                              min_r = 2,
                              max_r = 2)

#Create a random sat formula
n, r, m, formula = sat_gen.generate_formula()

print(f'n: {n}')
print(f'r: {r}')
print(f'm: {m}')
print(formula)


input_size = n #num of variables
embedding_size = n
hidden_size = 100
output_size = 2 #Two assignments: 0 or 1
num_layers = 2

accumulation_steps = 10
num_episodes = 100
clip_val = 1.0

policy_network = BasicRNN(input_size, embedding_size, hidden_size, output_size,
                          num_layers)

lr = 0.001
optimizer = optim.Adam(policy_network.parameters(), lr=lr)

history_loss, history_H = train(accumulation_steps, formula, num_episodes, clip_val)

print(loss)
'''