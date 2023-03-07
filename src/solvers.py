import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import src.train as train
from src.architecture.embeddings import GeneralEmbedding
from src.architecture.encoder_decoder import PolicyNetwork
from src.architecture.decoders import RNNDec, TransformerDec
from src.architecture.baselines import RolloutBaseline, EMABaseline

from src.initializers.var_initializer import BasicVar, Node2VecVar
from src.initializers.context_initializer import EmptyContext, Node2VecContext

from src.train import train
import src.utils as utils
from src.base_config import get_config


# from ray.air import session
from PyMiniSolvers import minisolvers

import os
import json
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

    # Node2vec embeddings
    if config['node2vec'] == False:
        n2v_emb = None

    elif config['node2vec'] == True:
        # Creates the folder n2v_dir/n2v_dim
        node2vec_dir = os.path.join(config['n2v_dir'], str(config['n2v_dim']))
        os.makedirs(node2vec_dir, exist_ok=True)

        # Creates a filename with the same name of the dimacs file but with extention .pt
        tail = os.path.split(config['data_dir'])[1]  # returns filename and extension
        node2vec_filename = os.path.splitext(tail)[0]  # returns filename
        
        node2vec_file = os.path.join(node2vec_dir, node2vec_filename + ".pt")

        # Tries to load pretrained embeddings
        n2v_emb = None
        if config['n2v_pretrained']:
            if os.path.isfile(node2vec_file):
                n2v_emb = torch.load(node2vec_file)
                print(f"Node2Vec embeddings of size {config['n2v_dim']} loaded from: {node2vec_file}.")
            else:
                print(f"No Node2Vec embeddings of size {config['n2v_dim']} have been created.")
        
        # Runs node2vec algorithm if not pretrained or not found
        if n2v_emb is None:
            n2v_emb = utils.node2vec(dimacs_path=config['data_dir'],
                                     device=device,
                                     embedding_dim=config['n2v_dim'],
                                     walk_length=config['n2v_walk_len'],
                                     context_size=config['n2v_context_size'],
                                     walks_per_node=config['n2v_walks_per_node'],
                                     p=config['n2v_p'],
                                     q=config['n2v_q'],
                                     batch_size=config['n2v_batch_size'],
                                     lr=config['n2v_lr'],
                                     num_epochs=config['n2v_num_epochs'],
                                     save_path=node2vec_dir,
                                     file_name=node2vec_filename,
                                     num_workers=config['n2v_workers'],
                                     verbose=config['n2v_verbose'])

    else:
        raise ValueError(f"{config['node2vec']} is not a valid value, try with True or False.")
    

    # Initializers
    # Var Initializers
    if config["dec_var_initializer"] == "BasicVar":
        initialize_dec_var  = BasicVar()
    elif config["dec_var_initializer"] == "Node2VecVar":
        if not config['node2vec']:
            raise ValueError("Node2VecVar variable initializer needs `config['node2vec']` set to True.")
        initialize_dec_var  = Node2VecVar()
    else:
        raise ValueError(f'{config["dec_var_initializer"]} is not a valid variable initializer.')
    
    # Context Initializers
    if config["dec_context_initializer"] == "EmptyContext":
        initialize_dec_context  = EmptyContext()
    elif config["dec_context_initializer"] == "Node2VecContext":
        if not config['node2vec']:
            raise ValueError("Node2VecContext context initializer needs `config['node2vec']` set to True.")
        initialize_dec_context  = Node2VecContext()
    else:
        raise ValueError(f'{config["dec_context_initializer"]} is not a valid context initializer.')
    
    # Embedding
    variable_size = num_variables if config["dec_var_initializer"] == "BasicVar" else config['n2v_dim'] * 2
    context_size = 0 if config["dec_context_initializer"] == "EmptyContext" else config['n2v_dim'] * 2
    if context_size == 0:
        # if context is empty, context_emb_size is 0.
        config['context_emb_size'] = 0
    emb_module = GeneralEmbedding(variable_size= variable_size,
                                  variable_emb_size=config['var_emb_size'],
                                  assignment_size=3,  # an assignment_t could be 0, 1 or 2 (SOS)
                                  assignment_emb_size=config['assignment_emb_size'],  
                                  context_size=context_size,
                                  context_emb_size=config['context_emb_size'],
                                  out_emb_size=config['model_dim'])


    # Decoder
    if (config["decoder"] == "GRU") or (config["decoder"] == "LSTM"):
        decoder = RNNDec(input_size = config['model_dim'],
                         cell = config["decoder"],
                         hidden_size = config['hidden_size'],
                         num_layers = config['num_layers'],
                         dropout = config['dropout'],
                         trainable_state=config["trainable_state"],
                         output_size = config['output_size'])
    

    elif config["decoder"] == "Transformer":
        decoder = TransformerDec(d_model = config['model_dim'],
                                num_heads = config['num_heads'],
                                num_layers = config['num_layers'],
                                dense_size = config['dense_size'],
                                dropout = config['dropout'],
                                activation = "relu",
                                output_size = config['output_size'])

    else:
        raise ValueError(f'{config["decoder"]} is not a valid decoder.')

    # Network
    policy_network = PolicyNetwork(formula=formula,
                                   num_variables=num_variables,
                                   n2v_emb=n2v_emb,
                                   emb_module=emb_module,
                                   decoder=decoder,  
                                   dec_var_initializer=initialize_dec_var,
                                   dec_context_initializer=initialize_dec_context)
    
    print("\n")
    utils.params_summary(policy_network)

    print(policy_network)
    
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
           
    if config['baseline'] is None:
        baseline = None
    elif config['baseline'] == 'ema':
        baseline = EMABaseline(num_clauses=m, alpha=config['alpha_ema'])
    elif config['baseline'] == 'greedy':
        baseline = RolloutBaseline(num_rollouts=-1)
    elif config['baseline'] == 'sample':
        if (type(config['k_samples']) != int) or (config['k_samples'] < 1):
            raise ValueError(f"`k_samples` must be an integer equeal or greater than 1.")
        baseline = RolloutBaseline(num_rollouts=config['k_samples'])
    else:
        raise ValueError(f'{config["baseline"]} is not a valid baseline.')
    
    active_search = train(formula= formula,
                          num_variables=num_variables,
                          policy_network=policy_network,
                          optimizer=optimizer,
                          device=device,
                          batch_size=config['batch_size'],
                          permute_vars = config['permute_vars'],
                          permute_seed = config['permute_seed'],
                          baseline = baseline,
                          logit_clipping=config['logit_clipping'],  
                          logit_temp=config['logit_temp'], 
                          entropy_weight = config['entropy_weight'],
                          clip_grad = config['clip_grad'],
                          num_episodes = config['num_episodes'],
                          accumulation_episodes = config['accumulation_episodes'],
                          log_interval = config['log_interval'],
                          eval_interval = config['eval_interval'],
                          eval_strategies = config['eval_strategies'], # 0 for greedy, i < 0 takes i samples and returns the best one.
                          writer = writer,  # Tensorboard writer
                          extra_logging = config['extra_logging'],
                          raytune = config['raytune'],
                          run_name = f"{config['run_name']}-{config['run_id']}",
                          progress_bar = config['progress_bar'],
                          early_stopping=config['early_stopping'], 
                          patience=config['patience'],
                          entropy_value=config['entropy_value'])

    # Saving best solution
    with open(os.path.join(config['save_dir'], "solution.json"), 'w') as f:
        json.dump(active_search, f, indent=True)

    if config['tensorboard_on']:
       writer.close()