import utils
import train
import eval
import torch.optim as optim

def random_solver(n, formula):
    # Create a random assignment
    assignment = utils.random_assignment(n=n)
    # Verifying the number of satisfied clauses
    is_sat, num_sat, eval_formula = utils.assignment_verifier(formula, assignment)
    return num_sat


def learning_solver(n, formula, policy_network, baseline, lr, accumulation_steps,
        num_episodes, entropy_weight, clip_logits, clip_val):
    optimizer = optim.Adam(policy_network.parameters(), lr=lr)
    history_loss, history_num_sat = train.train(accumulation_steps, formula, n, num_episodes,
                                                policy_network, optimizer, baseline, entropy_weight, clip_logits, clip_val, verbose=0)
    
    assignment = eval.eval(policy_network, num_variables=n)
    # Verifying the number of satisfied clauses
    is_sat, num_sat, eval_formula = utils.assignment_verifier(formula, assignment)
    return num_sat, history_loss, history_num_sat