import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseVar(nn.Module):
    """The base class for all variables initializers."""
    def __init__(self, *args, **kwargs):
        super(BaseVar, self).__init__(*args, **kwargs)
    
    def forward(self, enc_output, formula, num_variables, variables, *args):
        raise NotImplementedError


class BasicVar(BaseVar):
    """ Creates a tensor of integers from 0 to n-1 with shape [1, n, 1] """
    def __init__(self, *args, **kwargs):
        super(BasicVar, self).__init__(*args, **kwargs)
    
    def forward(self, enc_output, formula, num_variables, variables, *args):
        var = torch.tensor([i for i in range(num_variables)], dtype=torch.int64).reshape(1, -1, 1)
        #var = torch.cat([var] * batch_size)
        assert var.shape == (1, num_variables, 1), f"In BasicVar initializer. shape: {var.shape}, dtype: {var.dtype}."
        # ::var:: [batch_size=1, seq_len=num_variables, feature_size=1]
        return var


class IdentityEncoderOutputVar(BaseVar):
    """ Creates a tensor of integers from 0 to n-1 with shape [1, n, 1] """
    def __init__(self, *args, **kwargs):
        super(IdentityEncoderOutputVar, self).__init__(*args, **kwargs)
    
    def forward(self, enc_output, formula, num_variables, variables, *args):
        return enc_output
        # ::var:: [batch_size=1, seq_len=num_variables, feature_size=1]