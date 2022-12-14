import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from src.architecture.embeddings import ProjEmbedding, OneHotProjEmbedding, IdentityEmbedding
from src.architecture.encoder_decoder import EncoderDecoder
from src.architecture.decoders import RNNDecoder

from src.initializers.var_initializer import BasicVar
from src.initializers.context_initializer import EmptyContext
from src.initializers.state_initializer import ZerosState, TrainableState

# from src.baselines import BaselineRollout




# from src.train import train
import src.utils as utils
from src.base_config import get_config


from ray.air import session
from PyMiniSolvers import minisolvers

import os
import pprint as pp


def random_solver(n, formula):
    # Create a random assignment
    assignment = utils.random_assignment(n=n)
    # Verifying the number of satisfied clauses
    is_sat, num_sat, eval_formula = utils.num_sat_clauses(formula, assignment)
    return assignment, num_sat


def minisat_solver(n, formula):
    S = minisolvers.MinisatSolver()
    for i in range(n):
        S.new_var()
    for clause in formula:
        S.add_clause(clause)
    assignment = None
    is_sat = S.solve()
    if is_sat:
        assignment = list(S.get_model())
        #is_sat, num_sat, eval_formula = utils.num_sat_clauses(formula, assignment)
    return assignment, is_sat
        



# log_dir = 'logs_tb'

# run_id = time.strftime("%Y%m%dT%H%M%S")
# run_name = 'n' + str(num_variables)

# if tensorboard_on:
#     #log_dir = './outputs/' + experiment_name + '/runs/n' + str(num_variables) +'/'+str(r)
#     #log_dir = './outputs/' + runname
#     writer = SummaryWriter(log_dir = os.path.join(log_dir, exp_name, run_id + run_name))


def get_var_initializer(var_initializer):
    initial_var_class = {"BasicVar": BasicVar}.get(var_initializer, None)
    if initial_var_class is None:
        raise ValueError(f'{var_initializer} is not a valid variable initializer.')
    return initial_var_class


def get_context_initializer(context_initializer):
    initial_context_class = {"EmptyContext": EmptyContext}.get(context_initializer, None)
    if initial_context_class is None:
        raise ValueError(f'{context_initializer} is not a valid context initializer.')
    return initial_context_class


def get_state_initializer(state_initializer):
    initial_state_class = {"ZerosState": ZerosState,
                          "TrainableState": TrainableState}.get(state_initializer, None)
    if initial_state_class is None:
        raise ValueError(f'{state_initializer} is not a valid state initializer.')
    return initial_state_class


