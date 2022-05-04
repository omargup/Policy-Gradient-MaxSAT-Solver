import torch
import torch.distributions as distributions
import numpy as np

def params_summary(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)


def assignment_verifier(formula, assignment):
    '''Checks whether an assignment satisfies a CNF formula.
        Args:
            formula (list): CNF formula to be satisfied.
            assignment (list): Assignment to be verified.
                               Must be a list of 1s and 0s.
        Returns:
            is_sat (bool): True if assigment satisfies the
                           formula.
            num_sat (int): Number of satisfied clauses.
            eval_formula (list): List of 1s and 0s indicating
                                 which clauses were satisfied.
    '''
    eval_formula = []
    for clause in formula:
        eval_clause = []
        for literal in clause:
            if literal < 0:
                eval_literal = not assignment[abs(literal)-1]
            else:
                eval_literal = assignment[abs(literal)-1]
            eval_clause.append(eval_literal)
        eval_formula.append(any(eval_clause))
    is_sat = all(eval_formula)
    num_sat = sum(eval_formula)
    return is_sat, num_sat, eval_formula


def random_assignment(n):
    '''Creats a random assignment for a CNF formula.'''
    assignment = np.random.choice(2, size=n, replace=True)
    return assignment


def dimacs2list(dimacs_path):
    '''Reads a cnf file and returns the CNF formula
    in numpy format. 
        Args:
            dimacs_file (str): path to the DIMACS file.
        Returns:
            n (int): Number of variables in the formula.
            m (int): Number of clauses in the formula.
            formula (list): CNF formula.
    '''
    with open(dimacs_path, 'r') as f:
        dimacs = f.read()

    formula = []
    for line in dimacs.split('\n'):
        line = line.split()
        if line[0] == 'p':
            n = int(line[2])
            m = int(line[3])
        if len(line) != 0 and line[0] not in ("p","c","%"):
            clause = []
            for literal in line:
                literal = int(literal)
                if literal == 0:
                    break
                else:
                    clause.append(literal)
            formula.append(clause)
        elif line[0] == '%':
            break
    return n, m, formula


def greedy_strategy(action_logits):
    action = torch.argmax(action_logits, dim=-1)
    return action

def sampled_strategy(action_logits):
    action_dist = distributions.Categorical(logits= action_logits)
    #print(torch.nn.functional.softmax(action_logits, -1))  # for debugging
    action = action_dist.sample()
    return action

def sampling_assignment(formula, num_variables, variables, policy_network,
                        device, strategy):
    # Encoder
    enc_output = None
    if policy_network.encoder is not None:
        enc_output = policy_network.encoder(formula, num_variables, variables)

    # Initialize Decoder Variables 
    var = policy_network.init_dec_var(enc_output, formula, num_variables, variables)
    # ::var:: [batch_size, seq_len, feature_size]

    batch_size = var.shape[0]

    # Initialize action_prev at time t=0 with token 2.
    #   Token 0 is for assignment 0, token 1 for assignment 1
    action_prev = torch.tensor([2] * batch_size, dtype=torch.long).reshape(-1,1,1).to(device)
    # ::action_prev:: [batch_size, seq_len=1, feature_size=1]

    # Initialize Decoder Context
    context = policy_network.init_dec_context(enc_output, formula, num_variables, variables, batch_size).to(device)
    # ::context:: [batch_size, feature_size]

    # Initialize Decoder state
    state = policy_network.init_dec_state(enc_output, batch_size)
    if state is not None:
        state = state.to(device)
    # ::state:: [num_layers, batch_size=1, hidden_size]


    actions = []
    for t in range(num_variables):
        #TODO: send to device here.

        var_t = var[:,t:t+1,:].to(device)

        # Action logits
        action_logits, state= policy_network.decoder((var_t, action_prev, context), state)
        # ::action_logits:: [batch_size=1, seq_len=1, feature_size=2]
        #print(action_logits)  # for debugging

        if strategy == 'greedy':
            action =  greedy_strategy(action_logits)
        elif strategy == 'sampled':
            action = sampled_strategy(action_logits)
        else:
            raise TypeError("{} is not a valid strategy, try with 'greedy' or 'sampled'.".format(strategy))
        
        #print(action.item())  # for debugging
        actions.append(action.item())
        action_prev = action.unsqueeze(dim=-1)
        #::action_prev:: [batch_size, seq_len=1, feature_size=1]
        
    return actions

###################################################
# Neural network utils
###################################################

import torch.nn as nn
import torch_geometric.nn as gnn
from typing import List, Optional, Tuple, Union

def build_gcn_sequential_model(
    embedding_size: int,
    hidden_sizes: List[int]=[16],
    intermediate_fns: Optional[List[List[Union[nn.Module, None]]]]=[
        [None],
        [nn.ReLU(), nn.Dropout(p=0.2)], 
        [nn.ReLU()]
    ]

):

    net_sizes = [None] + hidden_sizes + [embedding_size]

    if intermediate_fns is not None:
        if len(net_sizes) != len(intermediate_fns):
            raise ValueError(f"intermediate_fns must had size {len(net_sizes)}. Got {len(intermediate_fns)}")
    else:
        intermediate_fns = [None] * len(net_sizes)


    sequence = []

    for search_idx in range(len(net_sizes) - 1):
        
        if intermediate_fns is not None and len(intermediate_fns) > search_idx and  intermediate_fns[search_idx] is not None:
            for intermediate_fn in intermediate_fns[search_idx]:
                if intermediate_fn is not None:
                    sequence.append(
                        (intermediate_fn, "x -> x")
                    )
                    
        
        sequence.append(
            (gnn.SAGEConv((-1, -1), net_sizes[search_idx + 1]), "x, edge_index -> x")
        )

    if intermediate_fns[-1] is not None:
        for intermediate_fn in intermediate_fns[-1]:
            if intermediate_fn is not None:
                sequence.append(
                    (intermediate_fn, "x -> x")
                )

    return gnn.Sequential("x, edge_index", sequence)


def build_gcn_model(
    embedding_size: int,
    hidden_sizes: List[int]=[16],
    intermediate_fns: Optional[List[List[Union[nn.Module, None]]]]=[
        [None],
        [nn.ReLU(), nn.Dropout(p=0.2)], 
        [nn.ReLU()]
    ],
    node_types: List[str] = ["variable", "clause"],
    edge_types: List[Tuple[str, str, str]]=[
        ("variable", "exists_in", "clause"),
        ("clause", "contains", "variable"),
    ]
):

    metadata = (node_types, edge_types)
    model = build_gcn_sequential_model(
        embedding_size,
        hidden_sizes,
        intermediate_fns
    )
    model = gnn.to_hetero(model, metadata)

    return model