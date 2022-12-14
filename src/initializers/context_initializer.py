import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseContext(nn.Module):
    """The base class for all contexts initializers."""
    def __init__(self, *args, **kwargs):
        super(BaseContext, self).__init__(*args, **kwargs)

    def forward(self, enc_output, formula, num_variables, variables, batch_size, *args):
        raise NotImplementedError


class EmptyContext(BaseContext):
    """ Returns an empty context."""
    def __init__(self, *args, **kwargs):
        super(EmptyContext, self).__init__(*args, **kwargs)

    def forward(self, enc_output, formula, num_variables, variables, batch_size, *args):
        return torch.empty([batch_size, 0])
        # ::context:: [batch_size, feature_size=0]
  
        
class EncoderOutputContext(BaseContext):
    """ Returns an empty context."""
    def __init__(self, aggregation="mean", *args, **kwargs):
        super(EncoderOutputContext, self).__init__(*args, **kwargs)
        if aggregation not in ["mean", "sum", "max", "min"]:
            raise ValueError("Supported aggregations are 'mean', 'sum', 'max' or 'min'")
        self.aggregation=aggregation

    def forward(self, enc_output, formula, num_variables, variables, batch_size, *args):
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
        
    