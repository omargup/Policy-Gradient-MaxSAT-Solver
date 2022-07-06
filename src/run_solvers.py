import torch
import torch.optim as optim

from src.decoders import RNNDecoder
from src.encoder_decoder import EncoderDecoder
from src.embeddings import BasicEmbedding, IdentityEmbedding
from src.baselines import BaselineRollout
from src.init_states import TrainableState

from src.init_vars import BasicVar
from src.init_contexts import EmptyContext
from src.init_states import ZerosState

from src.train import train
import src.utils as utils

from PyMiniSolvers import minisolvers

import os


def random_solver(n, formula):
    # Create a random assignment
    assignment = utils.random_assignment(n=n)
    # Verifying the number of satisfied clauses
    is_sat, num_sat, eval_formula = utils.assignment_verifier(formula, assignment)
    return num_sat


def minisat_solver(n, formula):
    S = minisolvers.MinisatSolver()
    for i in range(n):
        S.new_var()
    for clause in formula:
        S.add_clause(clause)
    num_sat = 0
    if S.solve():
        assignment = list(S.get_model())
        is_sat, num_sat, eval_formula = utils.assignment_verifier(formula, assignment)
    return num_sat
        

def learning_solver(config, checkpoint_dir=None, data_dir=None):

    # Device  
    device = 'cpu'
    if config['gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data
    if data_dir is None:
        raise ValueError("data_dir can not be None.") 
    
    n, m, formula = utils.dimacs2list(dimacs_path = data_dir)
    num_variables = n
    
    # Embeddings
    assignment_emb = BasicEmbedding(num_labels=3,
                                    embedding_size=config['embedding_size'])
    variable_emb = BasicEmbedding(num_labels=num_variables,
                                  embedding_size=config['embedding_size'])
    context_emb = IdentityEmbedding()

    # Encoder
    encoder = None

    # Decoder
    input_size = config['embedding_size'] * 2
    clip_logits_c = 0  # (default: 0)
    decoder = RNNDecoder(input_size = input_size,
                        cell = config['cell'],
                        assignment_emb = assignment_emb,
                        variable_emb = variable_emb,
                        context_emb = context_emb,
                        hidden_size = config['hidden_size'],
                        num_layers = config['num_layers'],
                        dropout = config['dropout'],
                        clip_logits_c = clip_logits_c)
    
    ## Initializers
    init_dec_var = BasicVar()
    init_dec_context = EmptyContext()
    init_dec_state = ZerosState()

    ## Network
    policy_network = EncoderDecoder(encoder=encoder,
                                    decoder=decoder,
                                    init_dec_var=init_dec_var,  # (None default)
                                    init_dec_context=init_dec_context,  # (None default)
                                    init_dec_state=init_dec_state)  # (None default)
    
    optimizer = optim.Adam(policy_network.parameters(), lr=config['lr'])
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        policy_network.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Training hyperparameters
    
    #TODO: clip_grad could be float or None
    #TODO: no dropout when layers==1

    variables = None

    baseline = None
    if config['baseline'] is not None:
        baseline = BaselineRollout(config['baseline'])

    history_loss, history_num_sat, hitosry_num_sat_val = train(formula,
                                                            num_variables,
                                                            variables,
                                                            config['num_episodes'],
                                                            config['accumulation_steps'],
                                                            policy_network,
                                                            optimizer,
                                                            device,
                                                            baseline,
                                                            config['entropy_weight'],
                                                            config['clip_grad'],
                                                            config['verbose'],
                                                            config['raytune'],
                                                            config['episode_log'],
                                                            config['log_step'])
    
    return history_loss, history_num_sat, hitosry_num_sat_val