import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEmbedding(nn.Module):
    """The base class for all Embeddings."""
    def __init__(self, *args, **kwargs):
        super(BaseEmbedding, self).__init__()
    
    def forward(self, X, *args):
        # X shape: [batch_size, seq_len, features_size]
        raise NotImplementedError

class ProjEmbedding(BaseEmbedding):
    """Applies a linear transformation"""
    def __init__(self, input_size, embedding_size=128, *args, **kwargs):
        super(ProjEmbedding, self).__init__()
        self.proj = nn.Linear(input_size, embedding_size)

    def forward(self, X):
        # X shape: [batch_size, seq_len, features_size=input_size]
        if X.dim() != 3:
            raise TypeError("X' shape must be [batch_size, seq_len, features_size].")
        return self.proj(X.float())
        # X shape: [batch_size, seq_len, features_size=embedding_size]


class OneHotProjEmbedding(BaseEmbedding):
    """Computes One-Hot enconding and then applies a linear transformation.
    Input X must be an integer."""
    def __init__(self, num_labels, embedding_size=128, *args, **kwargs):
        super(OneHotProjEmbedding, self).__init__()
        self.num_labels = num_labels
        self.proj = nn.Linear(num_labels, embedding_size)

    def forward(self, X):
        # X shape: [batch_size, seq_len, features_size=1]
        if X.dim() != 3 or X.shape[2] != 1 or X.dtype != torch.int64:
            raise TypeError("X must be a integer tensor (torch.int64) with shape [batch_size, seq_len, features_size=1].")
        X = X.squeeze(dim=-1)
        # X shape: [batch_size, seq_len]
        X = F.one_hot(X, self.num_labels).float()
        # X shape: [batch_size, seq_len, features_size=num_labels]
        return self.proj(X)
        # X shape: [batch_size, seq_len, features_size=embedding_size]


class IdentityEmbedding(BaseEmbedding):
    """ """
    def __init__(self, *args, **kwargs):
        super(IdentityEmbedding, self).__init__()

    def forward(self, X):
        # X shape: [batch_size, seq_len, features_size]
        if X.dim() != 3:
            if X.dim() != 2: 
                #TODO: valiedate input shape when context is empty
                raise TypeError("X must be a tensor with shape [batch_size, seq_len, features_size].")
        return X
        # X shape: [batch_size, seq_len, features_size]


# class OneHotEmbedding(Embedding):
#     """Computes One-Hot encoding."""

#     def forward(self, X):
#         # X shape: [batch_size, seq_len]
#         if X.dim() != 2:
#             raise TypeError("X must be a tensor with shape [batch_size, seq_len].")
#         X = F.one_hot(X, self.num_labels).float()
#         # ::x:: [batch_size, seq_len=1, num_features=num_labels]
#         return X


# class EmbeddingForLiterals(Embedding):
#     """ """
#     def __init__(self, num_labels, embedding_size, **kwargs):
#         super(EmbeddingForLiterals, self).__init__(**kwargs)
#         self.num_labels = num_labels
#         self.embedding = nn.Linear(num_labels, embedding_size)

#     def forward(self, X):
#         #X: [batch_size, seq_len=1]
#         if X < 0:
#             X = X + n
#         elif X > 0:
#             X = X + n - 1
    
#         X = F.one_hot(X, self.num_labels).float()
#         #x: [batch_size, seq_len=1, num_features=num_labels]
#         return self.embedding(X)
#         #x: [batch_size, seq_len=1, num_features=embedding_size]
#         #[-n, ..., -1, 1, .., n]