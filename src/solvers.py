import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import src.train as train
from src.architecture.embeddings import ProjEmbedding, OneHotProjEmbedding, IdentityEmbedding
from src.architecture.encoder_decoder import EncoderDecoder
from src.architecture.decoders import RNNDecoder

from src.initializers.var_initializer import BasicVar
from src.initializers.context_initializer import EmptyContext
from src.initializers.state_initializer import ZerosState, TrainableState

# from src.baselines import BaselineRollout


from src.train import train
import src.utils as utils
from src.base_config import get_config


# from ray.air import session
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


def pg_solver(config):

    # Configuration parameters
    config = get_config(config)
    pp.pprint(config)

    # Tensorboard
    writer = None
    if config['tensorboard_on']:
        writer = SummaryWriter(log_dir = os.path.join(config['log_dir'], config['exp_name'], f"{config['run_name']}-{config['run_id']}"))
    
    # Device  
    device = 'cpu'
    if config['gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'n\Running on {device}.')
    
    # Data
    if config['data_dir'] is None:
        raise ValueError("`config[data_dir]` can not be None.") 
    n, m, formula = utils.dimacs2list(dimacs_path = config['data_dir'])
    num_variables = n
    print(f"Formula loaded from: {config['data_dir']}.")


    # Initializers
    initial_var_class = get_var_initializer(config["dec_var_initializer"])
    initialize_dec_var  = initial_var_class()

    initial_context_class = get_context_initializer(config["dec_context_initializer"])
    initialize_dec_context  = initial_context_class()

    initial_state_class = get_state_initializer(config["dec_state_initializer"])
    initialize_dec_state  = initial_state_class(cell=config["cell"],
                                             hidden_size=config["hidden_size"],
                                             num_layers=config["num_layers"],
                                             a=config["initial_state_a"],
                                             b=config["initial_state_b"])


    # Embeddings
    # Assignment embedding
    assignment_emb = OneHotProjEmbedding(num_labels=3, # an assignment_t could be 0, 1 or 2 (2 is our SOS)
                                        embedding_size=config['assignment_emb_size'])

    # Variable embedding
    if config["dec_var_initializer"] == "BasicVar":
        variable_emb = OneHotProjEmbedding(num_labels=num_variables,
                                           embedding_size=config['variable_emb_size'])
    else:
        variable_emb = IdentityEmbedding()

    # Context embedding
    context_emb = IdentityEmbedding()


    # Encoder
    #TODO: code to handle more encoders
    encoder = None


    # Decoder
    #TODO: code to handle more decoders
    # if context is empty, context_emb_size is 0.
    input_size = config['variable_emb_size'] + config['assignment_emb_size'] \
        + config['context_emb_size']

    decoder = RNNDecoder(input_size = input_size,
                        cell = config['cell'],
                        assignment_emb = assignment_emb,
                        variable_emb = variable_emb,
                        context_emb = context_emb,
                        hidden_size = config['hidden_size'],
                        num_layers = config['num_layers'],
                        dropout = config['dropout'],
                        clip_logits_c = config['clip_logits_c'],
                        output_size = config['output_size'])
    
    # Network
    policy_network = EncoderDecoder(encoder=encoder,
                                    decoder=decoder,
                                    dec_var_initializer=initialize_dec_var,
                                    dec_context_initializer=initialize_dec_context,
                                    dec_state_initializer=initialize_dec_state)
    
    optimizer = optim.Adam(policy_network.parameters(), lr=config['lr'])

#     if config['raytune']:
#         loaded_checkpoint = session.get_checkpoint()
#         if loaded_checkpoint:
#             print("Loading from checkpoint.")
#             with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
#                 path = os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
#                 model_state, optimizer_state = torch.load(path)
#                 policy_network.load_state_dict(model_state)
#                 optimizer.load_state_dict(optimizer_state)
           

    #TODO: no dropout when layers==1

    variables = None

    baseline = None
    #if config['baseline'] is not None:
    #    baseline = BaselineRollout(config['baseline'])

    
    train(formula= formula,
          num_variables=num_variables,
          variables=variables,
          policy_network=policy_network,
          optimizer=optimizer,
          device=device,
          strategy='sampled',
          batch_size=config['batch_size'],
          permute_vars = config['permute_vars'],
          permute_seed = config['permute_seed'],
          baseline = baseline,
          entropy_weight = config['entropy_weight'],
          clip_grad = config['clip_grad'],
          raytune = config['raytune'],
          num_episodes = config['num_episodes'],
          accumulation_episodes = config['accumulation_episodes'],
          log_episodes = config['log_episodes'],
          eval_episodes = config['eval_episodes'],
          eval_strategies = config['eval_strategies'], # 0 for greedy, i < 0 takes i samples and returns the best one.
          writer = writer,  # Tensorboard writer
          extra_logging = config['extra_logging'],
          run_name = f"{config['run_name']}-{config['run_id']}",
          progress_bar = False)

    if config['tensorboard_on']:
       writer.close()