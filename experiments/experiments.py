from generator import UniformCNFGenerator
from architectures import BasicRNN
from architectures import BaselineRollout
import utils
import train
import eval

import torch
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt


def experiment1():
    num_experiments = 1 #num of times each experiment is run

    #CNF Formula
    n = 20 #Num variables
    k = 5  #Clause size
    r_list = [4.0,4.1,4.2,4.3,4.4] #Radious
    
    #Network
    embedding_size = 32
    hidden_size = 128
    num_layers = 1
    cell = 'GRU'
    dropout = 0
    input_size = n
    output_size = 2 #Two assignments: 0 or 1
    num_rollouts = 2 #For baseline

    #Training
    lr = 0.001
    accumulation_steps = 2
    num_episodes = 1500
    clip_val = 1

    
    num_sat_random = []
    #num_sat_rnn = []
    #num_sat_rnn_sb = []
    #num_sat_rnn_gb = []

    for _, r in enumerate(r_list):
        # Create a sat generator
        sat_gen = UniformCNFGenerator(min_n = n,
                                      max_n = n,
                                      min_k = k,
                                      max_k = k,
                                      min_r = r,
                                      max_r = r)

        # Create a random sat formula
        n, r, m, formula = sat_gen.generate_formula()

        ##################################################################
        # Random Model                                                   #
        ##################################################################
        num_sat = 0
        for i in range(num_experiments):
            # Create a random assignment
            assignment = utils.random_assignment(n=n)
            # Verifying whether the assinment satisfied the formula
            is_sat, num_sat, eval_formula = utils.assignment_verifier(formula, assignment)
            print(num_sat)
            num_sat += num_sat
            print(num_sat)
        num_sat_random.append(num_sat/num_experiments)


        # ##################################################################
        # # RNN Model                                                      #
        # ##################################################################
        # num_sat = 0
        # for i in range(num_experiments):
        #     # RNN model
        #     policy_network = BasicRNN(cell, input_size, embedding_size, hidden_size, output_size, num_layers, dropout)
        #     #baseline = BaselineRollout(num_rollouts, sampled)
        #     baseline = None
        #     optimizer = optim.Adam(policy_network.parameters(), lr=lr)
        #     history_loss, history_num_sat = train.train(accumulation_steps, formula, input_size, num_episodes,
        #                                                        policy_network, optimizer, baseline, clip_val)
        #     assignment = eval.eval(policy_network, num_variables=n)
        #     # Verifying whether the assinment satisfied the formula
        #     is_sat, num_sat, eval_formula = utils.assignment_verifier(formula, assignment)
        #     num_sat += num_sat
        # num_sat_rnn.append(num_sat/num_experiments)


        # ##################################################################
        # # RNN with sampled baseline                                      #
        # ##################################################################
        # num_sat = 0
        # for i in range(num_experiments):
        #     # RNN model
        #     policy_network = BasicRNN(cell, input_size, embedding_size, hidden_size, output_size, num_layers, dropout)
        #     baseline = BaselineRollout(num_rollouts, sampled=True)
        #     optimizer = optim.Adam(policy_network.parameters(), lr=lr)
        #     history_loss, history_num_sat = train.train(accumulation_steps, formula, input_size, num_episodes,
        #                                                        policy_network, optimizer, baseline, clip_val)
        #     assignment = eval.eval(policy_network, num_variables=n)
        #     # Verifying whether the assinment satisfied the formula
        #     is_sat, num_sat, eval_formula = utils.assignment_verifier(formula, assignment)
        #     num_sat += num_sat
        # num_sat_rnn_sb.append(num_sat/num_experiments)


        # ##################################################################
        # # RNN with greedy baseline                                      #
        # ##################################################################
        # num_sat = 0
        # for i in range(num_experiments):
        #     # RNN model
        #     policy_network = BasicRNN(cell, input_size, embedding_size, hidden_size, output_size, num_layers, dropout)
        #     baseline = BaselineRollout(num_rollouts, sampled=False)
        #     optimizer = optim.Adam(policy_network.parameters(), lr=lr)
        #     history_loss, history_num_sat = train.train(accumulation_steps, formula, input_size, num_episodes,
        #                                                        policy_network, optimizer, baseline, clip_val)
        #     assignment = eval.eval(policy_network, num_variables=n)
        #     # Verifying whether the assinment satisfied the formula
        #     is_sat, num_sat, eval_formula = utils.assignment_verifier(formula, assignment)
        #     num_sat += num_sat
        # num_sat_rnn_gb.append(num_sat/num_experiments)
    
    return num_sat_random #, num_sat_rnn, num_sat_rnn_sb, num_sat_rnn_gb


num_sat_random = experiment1()