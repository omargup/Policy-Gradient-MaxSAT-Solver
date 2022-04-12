import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseContext(nn.Module):
    """The base class for all contexts initializers."""
    def __init__(self, **kwargs):
        super(BaseContext, self).__init__(**kwargs)

    def forward(self, *args):
        raise NotImplementedError


class EmptyContext(BaseContext):
    """ Returns an empty context."""
    def __init__(self, **kwargs):
        super(EmptyContext, self).__init__(**kwargs)

    def forward(self, enc_output, formula, num_variables, variables, batch_size, *args):
        return torch.empty([batch_size, 0])
        # ::context:: [batch_size, feature_size=0]
        
    