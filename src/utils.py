import torch
import torch.distributions as distributions
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

import numpy as np
import networkx as nx
from sklearn.manifold import TSNE

from tbparse import SummaryReader
import seaborn as sns; sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
from PyMiniSolvers import minisolvers

#for node2vec
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
import os


def params_summary(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)


def num_sat_clauses_tensor(formula, assignment):
    '''Counts the number of clauses of a CNF formula that an assignment satisfies.
    The assignment is a torch tensor with shape [batch_size, num_variables].
    
    Arguments
    ---------
    -formula (list): A SAT formula in CNF.
    -assignment (tensor):: Assignment to be verified. Must be a torch tensor
                with shape [batch_size, num_variables]; e.g.: [[0,1,1], [1,0,1]].
   
    Returns
    -------
    -num_sat (tensor): Number of satisfied clauses with shape [batch_size].
    '''
    if not type(assignment) == torch.tensor:
        assert TypeError("'assignment' must be a tensor with shape [batch_size, num_variables]")

    batch_size = assignment.shape[0]
    num_sat_list = []
    for b in range(batch_size):
        _, num_sat, _ = num_sat_clauses(formula, assignment[b])
        num_sat_list.append(num_sat)
    return torch.tensor(num_sat_list, dtype=float)  # [batch_size]


def num_sat_clauses(formula, assignment):
    '''Counts the number of clauses of a CNF formula that an assignment satisfies.

    Arguments
    ---------
    -formula (list): A SAT formula in CNF.
    -assignment (list): Assignment to be verified.
                        Must be a list of 1s and 0s.
    
    Returns
    -------
    -sat (bool): True if assigment satisfies the formula, else False.
    -num_sat (int): Number of satisfied clauses.
    -eval_formula (list): List of 1s and 0s indicating which clauses were satisfied.
    '''
    # TODO: Verify len(list) == num_variables in formula
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


def assignment_eval(formula, assignment):
    """Uses PyMiniSolver to quickly check if an assignment satisfies or not a CNF formula. 

    Arguments
    ----------
    -formula (list): A SAT formula in CNF.
    -assignment (list): Assignment to be verified.
                        Must be a list of 1s and 0s.

    Returns
    ------
    -sat (bool): True if assigment satisfies the formula, else False.
    """
    # TODO: Verify if the assignment includes all variables.
    S = minisolvers.MinisatSolver()
    for _ in range(len(assignment)):
            S.new_var()
    for clause in formula:
        S.add_clause(clause)
    a = list(np.where(np.array(assignment)==1)[0] + np.array(1))
    return S.check_complete(positive_lits=a)


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
        elif len(line) != 0 and line[0] not in ("p","c","%"):
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


def dimacs2graph(dimacs_path):
    """
    Builds a graph from a dimacs file. The returned graph has the following elements:
        - 2n+m 1-dimensional nodes {0, 1, 2, ..., 2n+m-1} representing x_{0}, x_{1}, ..., x_{n-1}, -x_{0}, -x_{1},..., -x{n-1}, c_{0}, c_{1},..., c_{m-1}.
        - A set of edges between each literal and its negation, e.g.: the edge from x_{0} to -x_{0}.
        - Edges between literals and clauses.
    """

    # Load the formula in dimacs format and convert to list 
    n, m, formula = dimacs2list(dimacs_path = dimacs_path)

    # Nodes represented by a 2n+m x 1 tensor
    nodes = torch.tensor([[u] for u in range(2 * n + m)], dtype=torch.long)

    # Edges between literals and their negations
    edges = []
    for u in range(n):
        edges.append([u, u+n])
        edges.append([u+n, u])

    # Edges between litarals and clauses
    for c, clause in enumerate(formula):
        clause_node = 2 * n + c
        for lit in clause:
            literal_node = lit-1 if lit > 0 else -lit + n -1
            edges.append([clause_node, literal_node])
            edges.append([literal_node, clause_node])
    
    # Edges
    edge_index = torch.tensor(edges, dtype=torch.long)

    graph = Data(x=nodes,
                edge_index=edge_index.t().contiguous())

    graph.validate(raise_on_error=True)

    return n, m, graph


def graph2png(graph, n, m, filename, walk=None,
              seed=None, node_size=400, font_size=10):
    """
    Saves the plot of the graph representation of a SAT instance.
    """
    # TODO: add random_generator

    # Node names and types
    node_type = []
    node_name = []
    for u in range(n):
        node_type.append("p")
        node_name.append(f"x{u+1}")
    for u in range(n):
        node_type.append("n")
        node_name.append(f"-x{u+1}")
    for u in range(m):
        node_type.append("c")
        node_name.append(f"c{u+1}")
    
    G = to_networkx(data= graph, to_undirected= True)
    pos = nx.spring_layout(G, center=[0.5, 0.5])
    nx.set_node_attributes(G, pos, 'pos')
    nx.set_node_attributes(G, dict(zip(G.nodes(), node_type)), 'node_type')
    nx.set_node_attributes(G, dict(zip(G.nodes(), node_name)), 'node_name')


    fig = plt.figure(figsize=(10, 10), facecolor='white')
    color_state_map = {'p': "tab:green", 'n': 'tab:red', 'c': 'tab:blue'}

    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx_edges(G, pos,
                        width=1,
                        edge_color='tab:gray')

    if walk is not None:
        w = nx.path_graph(len(walk))
        nx.set_node_attributes(w, {idx: pos[node_id] for idx, node_id in enumerate(walk)}, 'pos')
        nx.draw_networkx_edges(w,
                               pos=nx.get_node_attributes(w, 'pos'),
                               width=6,
                               edge_color='tab:purple',
                               alpha=0.5)                         

    nx.draw_networkx_nodes(G, pos,
                        node_color='white',
                        node_size=node_size,
                        linewidths=1,
                        edgecolors=[color_state_map[node[1]['node_type']] 
                        for node in G.nodes(data=True)])

    nx.draw_networkx_nodes(G, pos,
                        node_color=[color_state_map[node[1]['node_type']] 
                        for node in G.nodes(data=True)],
                        alpha=0.3,
                        node_size=node_size)

    nx.draw_networkx_labels(G, pos,
                            labels=nx.get_node_attributes(G, 'node_name'),
                            font_size=font_size,
                            alpha=1.0,
                            font_color='black')
         
    plt.axis("off")
    fig.savefig(filename)


def n2v_train_epoch(model, loader, optimizer):
    model.train()
    epoch_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def node2vec(dimacs_path,
             device,
             embedding_dim=64,
             walk_length=20,
             context_size=10,
             walks_per_node=10,
             p=1,
             q=1,
             batch_size=32,
             lr=0.01,
             num_epochs=100,
             pretrained=True,
             save_dir='n2v_emb'):
    """
    Computes node2vec embeddings.
    If `pretrained`is True, tries to read in the `save_dir` directory a
    file with the precumputed embeddings, if `pretrained` is False or the file
    does not exists, node2vec embeddings are computed. Node embeddings
    are saved with the same name than the dimacs file but with extention '.pt'.
    """
    os.makedirs(save_dir, exist_ok=True)
    tail = os.path.split(dimacs_path)[1]  # returns filename and extension
    filename = os.path.splitext(tail)[0]  # returns filename
    emb_path = os.path.join(save_dir, filename + ".pt")
    
    if pretrained and os.path.isfile(emb_path):
        node_emb = torch.load(emb_path)
        if node_emb.shape[-1] != embedding_dim:
            raise ValueError(f'The embedding dimension ({node_emb.shape[-1]}) in the saved file  must the same than `embedding_dim` ({embedding_dim})')
        return node_emb

    print(f"Learning node2vec embeddings for the formula '{dimacs_path}'.")
    # Load the formula in dimacs format and convert to torch graph
    #n, m, graph = utils.dimacs2graph(dimacs_path=dimacs_path)
    n, m, graph = dimacs2graph(dimacs_path=dimacs_path)
    graph = graph.to(device)

    # Model definition
    model = Node2Vec(edge_index=graph.edge_index,
                     embedding_dim=embedding_dim,
                     walk_length=walk_length,
                     context_size=context_size,
                     walks_per_node=walks_per_node,
                     num_negative_samples=1,
                     p=p,
                     q=q,
                     sparse=True).to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    # Training
    for epoch in range(num_epochs):
        loss = n2v_train_epoch(model, loader, optimizer)
        print(f'Epoch: {epoch+1:02d}, Loss: {loss:.4f}')

    # Getting node embeddings
    with torch.no_grad():
        model.eval()
        embeddings = model(torch.arange(graph.num_nodes, device=device))
        # ::embeddings:: [seq_len=2n+m, feature_size=emb_dim]

    torch.save(embeddings, emb_path)
    node_emb = torch.load(emb_path)
    return node_emb


def node_emb2low_dim(node_embeddings, n, filename, dim=2, random_state=None):
    #TODO: Random state
    
    emb = TSNE(n_components=dim, init='pca', learning_rate='auto').fit_transform(node_embeddings.cpu().numpy())
    #y = graph.node_type  #.cpu().numpy()
    
    # Creating figure
    fig = plt.figure(figsize=(10, 10), facecolor='white')

    if dim==3:
        ax = fig.add_subplot(projection='3d')

        ax.scatter(emb[0:n, 0],
                   emb[0:n, 1],
                   emb[0:n, 2],
                   s=20, 
                   color='tab:green',
                   marker='o')
        
        ax.scatter(emb[n:2*n, 0],
                   emb[n:2*n, 1],
                   emb[n:2*n, 2],
                   s=20, 
                   color='tab:red',
                   marker='o')

        ax.scatter(emb[2*n:, 0],
                   emb[2*n:, 1],
                   emb[2*n:, 2],
                   s=20, 
                   color='tab:blue',
                   marker='o')
    
    elif dim==2:
        ax = fig.add_subplot()

        ax.scatter(emb[0:n, 0],
                   emb[0:n, 1],
                   s=30, 
                   color='tab:green',
                   alpha=0.5,
                   marker='o')
        
        ax.scatter(emb[n:2*n, 0],
                   emb[n:2*n, 1],
                   s=30, 
                   color='tab:red',
                   alpha=0.5,
                   marker='o')

        ax.scatter(emb[2*n:, 0],
                   emb[2*n:, 1],
                   s=30, 
                   color='tab:blue',
                   alpha=0.5,
                   marker='o')
        
    #plt.title("simple 3D scatter plot")
    #plt.axis('off')
    #plt.show()
    fig.savefig(filename)


class EntropyWeightDecay:
    """
    The first call to update_w(), returns `max` value, then the weight
    is decreasing. At step `steps`, the weight has the minimum `min` value
    and remains so.
    """
    def __init__(self, max, min, steps):
        self.min = float(min)
        self.max = float(max)
        self.step_size = (self.max - self.min) / float(steps-1)
        self.weight = self.max + self.step_size

    def update_w(self):
        self.weight = max(self.weight - self.step_size, self.min)
        return self.weight


#sns.choose_diverging_palette(as_cmap=False)
def probs2plot(log_dir, img_path):
    # Read tensorboard logs with tbparser
    reader = SummaryReader(log_dir, pivot=False)  # extra_columns={'dir_name'}
    df = reader.tensors

    # Keep only rows with tag == 'prob_1'
    df = df[df['tag'] == 'prob_1']
    # Keep only step and value columns
    df = df[['step', 'value']]

    # Get stats
    min_step = df['step'].min()
    max_step = df['step'].max()
    num_episodes = df['step'].nunique()
    num_variables = df[df['step'] == min_step].shape[0]

    # For debugging
    #print(min_step)
    #print(max_step)
    #print(num_episodes)
    #print(num_variables)

    df.rename(columns = {'step': 'Episode'}, inplace = True)

    # Add an extra column with variable indices
    variables = []
    for _ in range(num_episodes):
        for i in range(num_variables):
            variables.append(i)
    df['Variable'] = variables

    # Reshape dataframe
    table = pd.pivot_table(df, index='Variable', columns='Episode', values='value')

    #colormap = sns.color_palette("coolwarm", as_cmap=True)
    colormap = sns.diverging_palette(259, 359, s=90, l=60, sep=5, as_cmap=True) 
    plt.figure(figsize = (20,8))
    ax = sns.heatmap(table, vmin=0, vmax=1, square=True,  xticklabels=5,
                    linewidths=0.0, cmap=colormap, cbar=True, cbar_kws={"shrink": .835, 'pad':0.02})
        #for i in range(0, 102 ,2):
        #    ax.axvline(i, color='white', lw=5)

    #ax.set(title='Variables assignments through episodes')
    #ax.set_xticklabels([i for i in range(100, 5000+1, 500)])
    plt.tight_layout()
    plt.savefig(img_path)
    plt.show()
    

def sampled_sol(formula, assignment):
    '''Computes the number of clauses that every assigment in the batch satiesfies and returns the best one.
        The formula must be in CNF format and the assignment is a torch
        tensor with shape [batch_size, num_variables].
        Arguments
        ---------
            formula (list): CNF formula to be satisfied.
            assignment (tensor): Assignment to be verified. Must be a torch tensor
                with shape [batch_size, num_variables]; e.g.: [[0,1,1], [1,0,1]].
        Returns
        -------
            num_sat (int): Average number of satisfied clauses.
    '''
    if not type(assignment) == torch.tensor:
        assert TypeError("'assignment' must be a tensor with shape [batch_size, num_variables]")

    batch_size = assignment.shape[0]
    total_num_sat = 0
    
    best_num_sat = 0
    best_assignment = None
    for b in range(batch_size):
        _, num_sat, _ = num_sat_clauses(formula, assignment[b])

        if num_sat > best_num_sat:
                best_num_sat = num_sat
                best_assignment = assignment[b]
    
    return best_num_sat, best_assignment





def random_assignment(n):
    '''Creats a random assignment for a CNF formula.'''
    assignment = np.random.choice(2, size=n, replace=True)
    return assignment





def greedy_strategy(action_logits):
    action = torch.argmax(action_logits, dim=-1)
    return action

def sampled_strategy(action_logits):
    action_dist = distributions.Categorical(logits= action_logits)
    action = action_dist.sample()
    return action

def sampling_assignment(formula,
                        num_variables,
                        variables,
                        policy_network,
                        device,
                        strategy,
                        batch_size = 1,
                        permute_vars = False):
    # Encoder
    enc_output = None
    if policy_network.encoder is not None:
        enc_output = policy_network.encoder(formula, num_variables, variables)

    # Initialize Decoder Variables 
    var = policy_network.init_dec_var(enc_output, formula, num_variables, variables, batch_size)
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
    # ::state:: [num_layers, batch_size, hidden_size]

    # Episode Buffer
    buffer_action = torch.empty(size=(batch_size, num_variables))
    # ::buffer_action:: [batch_size, seq_len=num_variables]

    if permute_vars:
        permutation = torch.cat([torch.randperm(num_variables).unsqueeze(0) for _ in range(batch_size)], dim=0).permute(1,0)
    else:
        permutation = torch.cat([torch.tensor([i for i in range(num_variables)]).unsqueeze(0) for _ in range(batch_size)], dim=0).permute(1,0)
        # ::permutation:: [num_variables, batch_size]

    idx = [i for i in range(batch_size)]
    
    for t in permutation:
        var_t = var[idx, t].unsqueeze(1).to(device)
        assert var_t.shape == (batch_size, 1, var.shape[-1])
        # ::var_t:: [batch_size, seq_len=1, feature_size]

        # Action logits
        action_logits, state = policy_network.decoder((var_t, action_prev, context), state)
        # ::action_logits:: [batch_size, seq_len=1, feature_size=2]

        if strategy == 'greedy':
            action =  greedy_strategy(action_logits)
        elif strategy == 'sampled':
            action = sampled_strategy(action_logits)
        else:
            raise TypeError("{} is not a valid strategy, try with 'greedy' or 'sampled'.".format(strategy))
        
        #buffer update
        buffer_action[idx, t] = action
        
        action_prev = action.unsqueeze(dim=-1)
        #::action_prev:: [batch_size, seq_len=1, feature_size=1]
    
        
    return buffer_action.detach().cpu().numpy()





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