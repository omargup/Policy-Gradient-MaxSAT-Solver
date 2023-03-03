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


class GeneralEmbedding(BaseEmbedding):
    """ First, applies a linear transormation to each input, then
    concatenates these transformation and aplies a final linear transformation.
    If `context_size` is 0, `context_emb_size` will be 0.
    """
    def __init__(self,
                 variable_size,
                 variable_emb_size,
                 assignment_size,
                 assignment_emb_size,
                 context_size,
                 context_emb_size,
                 out_emb_size,
                 *args, **kwargs):
        super(GeneralEmbedding, self).__init__()
        #self.out_emb_size = out_emb_size
        self.variable_proj = nn.Linear(variable_size, variable_emb_size)
        self.assignment_proj = nn.Linear(assignment_size, assignment_emb_size)
        self.context_size = context_size
        if context_size < 0:
            raise ValueError(f"`context_size` must be 0 or greater than 0, got {context_size}.")
        if context_size != 0:
            self.context_proj = nn.Linear(context_size, context_emb_size)
        if context_size == 0:
            context_emb_size = 0
        input_size = variable_emb_size + assignment_emb_size + context_emb_size    
        self.out_emb =  nn.Linear(input_size, out_emb_size)

    def forward(self, variable, assignment, context):
        # variable: [batch_size, seq_len, features_size], could be n or 2*n2v_emb_size
        # assignment: [batch_size, seq_len, features_size=3]
        # context: [batch_size, seq_len, features_size], feature_size could be 0 or 2*n2v_emb_size.

        assert (variable.dim() == 3) and (variable.dtype == torch.float), \
            f"In GeneralEmbedding's input. var dim: {variable.dim()}, shape: {variable.shape}, dtype: {variable.dtype}."
        assert (assignment.dim() == 3) and (assignment.dtype == torch.float), \
            f"In GeneralEmbedding's input. assignment dim: {assignment.dim()}, shape: {assignment.shape}, dtype: {assignment.dtype}."
        assert (context.dim() == 3) and (context.dtype == torch.float), \
            f"In GeneralEmbedding's input. context dim: {context.dim()}, var shape: {context.shape}, dtype: {context.dtype}."

        v = self.variable_proj(variable)
        a = self.assignment_proj(assignment)
        if self.context_size != 0:
            c = self.context_proj(context)
            cat_vec = torch.cat((v,a,c), dim=-1)
        else:
            cat_vec = torch.cat((v,a), dim=-1)
        emb = self.out_emb(cat_vec)

        return emb
        # emb: [batch_size, seq_len, features_size=output_emb]






class ProjEmbedding(BaseEmbedding):
    """Applies a linear transformation"""
    def __init__(self, input_size, embedding_size=128, *args, **kwargs):
        super(ProjEmbedding, self).__init__()
        self.proj = nn.Linear(input_size, embedding_size)

    def forward(self, X):
        # X shape: [batch_size, seq_len, features_size=input_size]
        assert (X.dim() == 3) and (X.dtype == torch.float), \
            f"In ProjEmbedding's input. dim: {X.dim()}, shape: {X.shape}, dtype: {X.dtype}."

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
        assert (X.dim() == 3) and (X.shape[-1] == 1) and (X.dtype == torch.int64), \
            f"In OneHotProjEmbedding's input. dim: {X.dim()}, shape: {X.shape}, dtype: {X.dtype}."
     
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
        # X shape: [batch_size, seq_len, features_size] or [batch_size, features_size]
        assert (X.dim() == 3) or (X.dim() == 2), \
            f"In IdentityEmbedding's input. dim: {X.dim()}, shape: {X.shape}."

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