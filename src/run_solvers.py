import torch.optim as optim

from src.encoder_decoder import EncoderDecoder
from src.train import train
import utils

from PyMiniSolvers import minisolvers

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
        

def learning_solver(formula, num_variables, variables, encoder, decoder, init_dec_var,
                    init_dec_context, init_dec_state, num_episodes, accumulation_steps,
                    lr, device, baseline, entropy_weight, clip_val, verbose):

    policy_network = EncoderDecoder(encoder=encoder,
                                    decoder=decoder,
                                    init_dec_var=init_dec_var,  # (None default)
                                    init_dec_context=init_dec_context,  # (None default)
                                    init_dec_state=init_dec_state)  # (None default)

    optimizer = optim.Adam(policy_network.parameters(), lr=lr)

    history_loss, history_num_sat = train(formula,
                                        num_variables,
                                        variables,
                                        num_episodes,
                                        accumulation_steps,
                                        policy_network,
                                        optimizer,
                                        device,
                                        baseline,
                                        entropy_weight,
                                        clip_val,
                                        verbose)
    
    strategy='greedy'
    assignment = utils.sampling_assignment(formula, num_variables, variables,
                                           policy_network, device, strategy)
    # Verifying the number of satisfied clauses
    is_sat, num_sat, eval_formula = utils.assignment_verifier(formula, assignment)
    return num_sat, history_loss, history_num_sat