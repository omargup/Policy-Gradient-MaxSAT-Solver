import torch.nn as nn
import torch.nn.functional as F


class BaseEmbedding(nn.Module):
    """The base class for all Embeddings."""
    def __init__(self, **kwargs):
        super(BaseEmbedding, self).__init__(**kwargs)
    
    def forward(self, X, *args):
        raise NotImplementedError


class BasicEmbedding(BaseEmbedding):
    """Computes One-Hot enconding and then applies a linear transformation"""
    def __init__(self, num_labels, embedding_size=128, **kwargs):
        super(BasicEmbedding, self).__init__(**kwargs)
        self.num_labels = num_labels
        self.embedding = nn.Linear(num_labels, embedding_size)

    def forward(self, X):
        # X shape: [batch_size, seq_len, features_size=1]
        if X.dim() != 3 or X.shape[2] != 1:
            raise TypeError("X must be a tensor with shape [batch_size, seq_len, features_size=1].")
        X = X.view(1,-1)
        # X shape: [batch_size, seq_len]
        X = F.one_hot(X, self.num_labels).float()
        # X shape: [batch_size, seq_len, num_features=num_labels]
        return self.embedding(X)
        # X shape: [batch_size, seq_len, num_features=embedding_size]


class IdentityEmbedding(BaseEmbedding):
    """ """
    def __init__(self, num_labels, embedding_size=128, **kwargs):
        super(IdentityEmbedding, self).__init__(**kwargs)

    def forward(self, X):
        # X shape: [batch_size, seq_len, features_size]
        if X.dim() != 3:
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