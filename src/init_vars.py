import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseVar(nn.Module):
    """The base class for all variables initializers."""
    def __init__(self, **kwargs):
        super(BaseVar, self).__init__(**kwargs)
    
    def forward(self, enc_output, formula, num_variables, variables, *args):
        raise NotImplementedError


class BasicVar(BaseVar):
    """ Creates a tensor of integers from 0 to n-1 with shape [1, n, 1] """
    def __init__(self, **kwargs):
        super(BasicVar, self).__init__(**kwargs)
    
    def forward(self, enc_output, formula, num_variables, variables, *args):
        return torch.tensor([i for i in range(num_variables)]).reshape(1,-1,1)
        # ::var:: [batch_size=1, seq_len=num_variables, feature_size=1]

class IdentityEncoderOutputVar(BaseVar):
    """ Creates a tensor of integers from 0 to n-1 with shape [1, n, 1] """
    def __init__(self, **kwargs):
        super(IdentityEncoderOutputVar, self).__init__(**kwargs)
    
    def forward(self, enc_output, formula, num_variables, variables, *args):
        return enc_output
        # ::var:: [batch_size=1, seq_len=num_variables, feature_size=1]