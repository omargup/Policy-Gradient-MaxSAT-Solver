import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import src.train as train
from src.architecture.embeddings import GeneralEmbedding
from src.architecture.encoder_decoder import PolicyNetwork
from src.architecture.decoders import RNNDec, TransformerDec
from src.architecture.baselines import RolloutBaseline, EMABaseline, ZeroBaseline

from src.initializers.var_initializer import BasicVar, Node2VecVar
from src.initializers.context_initializer import EmptyContext, Node2VecContext

from src.train import train
import src.utils as utils
from src.base_config import get_config

from ray import air, tune
from ray.tune.schedulers import ASHAScheduler

from ray.air import session
from PyMiniSolvers import minisolvers

import os
import json
import pprint as pp

from GPUtil import showUtilization as gpu_usage


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
    if not config["raytune"]:
        print("\n")
        pp.pprint(config)
    
    if config["raytune"]:
        config['tensorboard_on'] = False
        config['extra_logging'] = False
    
    # Saving configuration
    if not config["raytune"]:
        with open(os.path.join(config['save_dir'], "config.json"), 'w') as f:
            json.dump(config, f, indent=True)

    # Verbose
    if (config['verbose'] < 0) or (config['verbose'] > 2) or (type(config['verbose']) != int): 
        raise ValueError(f"`config['verbose']` must be 0, 1 or 2, got {config['verbose']}.") 

    # Tensorboard
    writer = None
    if config['tensorboard_on']:
        writer = SummaryWriter(config['log_dir'])
    
    # Device  
    device = 'cpu'
    if config['gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config['verbose'] > 0:
        print(f'\nRunning on {device}.')
    
    # Data
    if config['data_dir'] is None:
        raise ValueError("`config[data_dir]` can not be None.") 
    n, m, formula = utils.dimacs2list(dimacs_path = config['data_dir'])
    num_variables = n
    if config['verbose'] > 0:
        print(f"\nFormula loaded from: {config['data_dir']}.")

    # ###########################################################################
    # print("\nBefore load node2vec:", torch.cuda.memory_allocated(device))
    # print("\tAllocated:", round(torch.cuda.memory_allocated(device)/1024**3,1), "GB")
    # print("\tCached:", round(torch.cuda.memory_reserved(device)/1024**3,1), "GB") 
    # gpu_usage()
    # ###########################################################################
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
        # If raytune is set to True, pretrained embeddings must exist
        n2v_emb = None
        if config['n2v_pretrained']:
            if os.path.isfile(node2vec_file):
                n2v_emb = torch.load(node2vec_file, map_location=device)
                if config['verbose'] > 0:
                    print(f"\nNode2Vec embeddings of size {config['n2v_dim']} loaded from: {node2vec_file}.")
            else:
                if config["raytune"]:
                    raise Exception(f"No node2vec emb of size {config['n2v_dim']} was found at {node2vec_file}.")
                if config['verbose'] > 0:
                    print(f"\nNo Node2Vec embeddings of size {config['n2v_dim']} have been created yet.")

        # Runs node2vec algorithm if not pretrained or not found (raytune must be False)
        if (n2v_emb is None) and not config["raytune"]:
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
                                     raytune=False,
                                     verbose=config['n2v_verbose'])
            print(f"\nThe {config['n2v_dim']}-dim Node2Vec embeddings of this instance has been created.")

    else:
        raise ValueError(f"{config['node2vec']} is not a valid value, try with True or False.")
    
    # ###########################################################################
    # print("\n1. After load node2vec:", torch.cuda.memory_allocated(device), 'B')
    # print("\tAllocated:", round(torch.cuda.memory_allocated(device)/1024**3,1), "GB")
    # print("\tCached:", round(torch.cuda.memory_reserved(device)/1024**3,1), "GB")
    # gpu_usage()     
    # ###########################################################################
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
                                   n2v_emb=n2v_emb if n2v_emb is None else n2v_emb.to(device),
                                   emb_module=emb_module,
                                   decoder=decoder,  
                                   dec_var_initializer=initialize_dec_var,
                                   dec_context_initializer=initialize_dec_context)
    
    optimizer = optim.Adam(policy_network.parameters(), lr=config['lr'], maximize=True)

    #policy_network = torch.compile(model)
    
    # if config["raytune"]:
    #     loaded_checkpoint = session.get_checkpoint()
    #     if loaded_checkpoint:
    #         print("\nLoading from checkpoint...")
    #         with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
    #             path = os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
    #             model_state, optimizer_state = torch.load(path)
    #             policy_network.load_state_dict(model_state)
    #             optimizer.load_state_dict(optimizer_state)
           
    if config['baseline'] == 'zero':
        baseline = ZeroBaseline()
    elif config['baseline'] == 'ema':
        baseline = EMABaseline(num_clauses=m, alpha=config['alpha_ema'])
    elif config['baseline'] == 'greedy':
        baseline = RolloutBaseline(num_rollouts=-1)
    elif config['baseline'] == 'sample':
        if (type(config['k_samples']) != int) or (config['k_samples'] < 1):
            raise ValueError(f"`k_samples` must be an integer equeal or greater than 1 got {config['k_samples']} with type: {type(config['k_samples'])}.")
        baseline = RolloutBaseline(num_rollouts=config['k_samples'], temperature=config['sampling_temp'])
    else:
        raise ValueError(f'{config["baseline"]} is not a valid baseline.')
    
    # ###########################################################################
    # # Number of parameters
    # print("\t Params:", sum(p.numel() for p in policy_network.parameters()))
    # print("\t Trainable params:", sum(p.numel() for p in policy_network.parameters() if p.requires_grad))
    
    # print("2. After build the full model", torch.cuda.memory_allocated(device))
    # print("\tAllocated:", round(torch.cuda.memory_allocated(device)/1024**3,1), "GB")
    # print("\tCached:", round(torch.cuda.memory_reserved(device)/1024**3,1), "GB")
    # gpu_usage() 
    # ###########################################################################
    active_search = train(formula=formula,
                          num_variables=num_variables,
                          policy_network=policy_network,
                          optimizer=optimizer,
                          device=device,
                          batch_size=config['batch_size'],
                          permute_vars = config['permute_vars'],
                          permute_seed = config['permute_seed'],
                          baseline = baseline,
                          logit_clipping=config['logit_clipping'],
                          #logit_temp=config['logit_temp'],
                          entropy_estimator=config['entropy_estimator'],
                          beta_entropy = config['beta_entropy'],
                          clip_grad = config['clip_grad'],
                          num_samples = config['num_samples'],
                          accumulation_episodes = config['accumulation_episodes'],
                          log_interval = config['log_interval'],
                          eval_interval = config['eval_interval'],
                          eval_strategies = config['eval_strategies'],
                          writer = writer,
                          extra_logging = config['extra_logging'],
                          raytune = config['raytune'],
                          run_name = f"{config['run_name']}-{config['run_id']}",
                          save_dir = config['save_dir'],
                          sat_stopping=config['sat_stopping'],
                          verbose = config['verbose'],
                          checkpoint_dir=config['checkpoint_dir'])

    if config['tensorboard_on']:
       writer.close()