import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architecture.embeddings import BaseEmbedding
#from typing import List, Optional, Tuple, Union
#from src.utils import build_gcn_model
#from torch_geometric.data import HeteroData


class Encoder(nn.Module):
    """ """
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class RNNEncoder(Encoder):
    """ """
    def __init__(self,
                 cell='GRU',
                 embedding=None,
                 embedding_size=128,
                 hidden_size=128,
                 num_layers=1,
                 dropout=0, **kwargs):
        super(RNNEncoder, self).__init__(**kwargs)

        # Embedding
        self.embedding = embedding
        if embedding is None:
            raise TypeError("An embedding must be specified.")
        elif embedding is not None:
            if not issubclass(type(self.embedding), BaseEmbedding):
                raise TypeError("embedding must inherit from BaseEmbedding.")
        
        # RNN
        if cell == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout = dropout)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout)
        else:
            raise TypeError("{} is not a valid cell, try with 'LSTM' or 'GRU'.".format(self.cell))

    def forward(self, formula, *args):
        # ::formula:: [batch_size, seq_len, features_size]
        X = self.embedding(formula)
        # ::X:: [batch_size, seq_len, embedding_size]
        X = X.permute(1, 0, 2)
        # ::X:: [seq_len, batch_size, embedding_size]
        output, state = self.rnn(X)  # Initial state is zeros
        # ::output:: [seq_len, batch_size, hidden_size]
        # ::state:: [num_layers, batch_size, hidden_size]
        return output, state


# class GCNEncoder(Encoder):
#     def __init__(self,
#         embedding_size: int,
#         hidden_sizes: List[int]=[16],
#         intermediate_fns: Optional[List[List[Union[nn.Module, None]]]]=[
#             [None],
#             [nn.ReLU(), nn.Dropout(p=0.2)], 
#             [nn.ReLU()]
#         ],
#         device="cpu",
#         **kwargs
#     ):

#         super(GCNEncoder, self).__init__(**kwargs)

#         self.device = device

#         self.module_ = build_gcn_model(
#             embedding_size,
#             hidden_sizes=hidden_sizes,
#             intermediate_fns=intermediate_fns,
#         )

#     # def forward(self, x, edge_index):    
#     def forward(self, formula, num_variables, variables):
#         #print("Formula", formula)
#         #print("Num variables", num_variables)
#         #print("Variables", variables)

#         # literals shape: [1, num_variables]
#         # literals = F.one_hot(torch.arange(num_variables).unsqueeze(0), num_variables)
#         literals = torch.eye(num_variables, dtype=torch.float)
#         clauses = torch.tensor(formula, dtype=torch.float)

#         edges = []

#         for clause_idx, clause in enumerate(formula):
#             for literal in clause:
#                 edges.append([abs(literal) - 1, clause_idx])

#         edges = torch.tensor(edges, dtype=torch.long)

#         data = HeteroData()
#         data["variable"].x = literals.to(self.device)
#         data["clause"].x = clauses.to(self.device)

#         data["variable", "exists_in", "clause"].edge_index = torch.clone(edges).T.contiguous().to(self.device)
#         data["clause", "contains", "variable"].edge_index = torch.clone(edges[:, [1, 0]]).T.contiguous().to(self.device)

#         out = self.module_(data.x_dict, data.edge_index_dict)

#         return out["variable"].unsqueeze(0)




