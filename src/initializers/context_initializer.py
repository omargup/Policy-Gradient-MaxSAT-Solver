import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseContext(nn.Module):
    """The base class for all contexts initializers."""
    def __init__(self, *args, **kwargs):
        super(BaseContext, self).__init__()

    def forward(self, enc_output, formula, num_variables, *args):
        raise NotImplementedError


class EmptyContext(BaseContext):
    """ Returns an empty context."""
    def __init__(self, *args, **kwargs):
        super(EmptyContext, self).__init__()

    def forward(self, enc_output, formula, num_variables, *args):
        return torch.empty([1, 0])
        # ::context:: [batch_size=1, feature_size=0]
  

class Node2VecContext(BaseContext):
    """ Returns a context from node2vec encoder's output."""
    def __init__(self, *args, **kwargs):
        super(Node2VecContext, self).__init__()

    def forward(self, enc_output, formula, num_variables, *args):
        # ::enc_output:: [seq_len=2n+m, feature_size=n2v_emb_dim]
        literals = enc_output[:2*num_variables]
        clauses = enc_output[2*num_variables:]
        lit_avg = literals.mean(dim=0)
        cla_avg = clauses.mean(dim=0)
        context = torch.cat((lit_avg, cla_avg))
        context = context.unsqueeze(0) 
        return context
        # ::context:: [batch_size=1, feature_size=2*n2v_emb_dim]
        





class EncoderOutputContext(BaseContext):
    """ Returns an empty context."""
    def __init__(self, aggregation="mean", *args, **kwargs):
        super(EncoderOutputContext, self).__init__()
        if aggregation not in ["mean", "sum", "max", "min"]:
            raise ValueError("Supported aggregations are 'mean', 'sum', 'max' or 'min'")
        self.aggregation=aggregation

    def forward(self, enc_output, formula, num_variables, *args):
        if self.aggregation == "mean":
            return enc_output.mean(dim=-2)
        elif self.aggregation == "sum":
            return enc_output.sum(dim=-2)
        elif self.aggregation == "max":
            return enc_output.max(dim=-2)[0]
        elif self.aggregation == "min":
            return enc_output.min(dim=-2)[0]
        else:
            raise ValueError(f"Aggregation '{self.aggregation}' not supported")
        # ::context:: [batch_size, feature_size=0]
        
    