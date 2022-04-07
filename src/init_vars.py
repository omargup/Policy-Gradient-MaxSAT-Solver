import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseVars(nn.Module):
    """The base class for all variables initializers."""
    def __init__(self, **kwargs):
        super(BaseVars, self).__init__(**kwargs)
    
    def forward(self, *args):
        raise NotImplementedError


class BasicVars(BaseVars):
    """ Creates a tensor of integers from 0 to n-1 with shape [1, n, 1] """
    def __init__(self, **kwargs):
        super(BasicVars, self).__init__(**kwargs)
    
    def forward(self, enc_output, formula, num_variables, variables, device, *args):
        return torch.tensor([i for i in range(num_variables)], device=device).reshape(1,-1,1)
        # ::var:: [batch_size, seq_len, feature_size]