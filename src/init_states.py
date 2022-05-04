import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseState(nn.Module):
    """The base class for all initial states."""
    def __init__(self, **kwargs):
        super(BaseState, self).__init__(**kwargs)
    
    def forward(self, enc_output, batch_size, *args):
        raise NotImplementedError


class ZerosState(BaseState):
    """ """
    def __init__(self, **kwargs):
        super(ZerosState, self).__init__(**kwargs)
    
    def forward(self, enc_output, batch_size, *args):
        # h_0 defaults to zeros with the proper shape if initial
        # state is equeal to None
            return None

# class ZerosState(BaseState):
#     """ """
#     def __init__(self, cell, hidden_size, num_layers, **kwargs):
#         super(ZerosState, self).__init__(**kwargs)
#         self.cell = cell
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
    
#     def forward(self, enc_output, batch_size, *args):
#         if self.cell == 'GRU':
#             # state shape: [num_layers, batch_size, hidden_size]
#             return torch.zeros(self.num_layers, batch_size, self.hidden_size)

#         elif self.cell == 'LSTM':
#             # h shape: [num_layers, batch_size, hidden_size]
#             h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
#             # c shape: [num_layers, batch_size, hidden_size]
#             c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
#             return (h, c)


class TrainableState(BaseState):
    """The base class for all initial states."""
    def __init__(self, cell, hidden_size, num_layers, a=-0.8, b=0.8, **kwargs):
        super(TrainableState, self).__init__(**kwargs)
        self.cell = cell

        if cell == 'GRU':
            # state shape: [num_layers, batch_size=1, hidden_size]
            self.h = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
            nn.init.uniform_(self.h, a=a, b=b)
            
        
        elif cell == 'LSTM':
            # h shape: [num_layers, batch_size=1, hidden_size]
            self.h = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
            # c shape: [num_layers, batch_size=1, hidden_size]
            self.c = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
            nn.init.uniform_(self.h, a=a, b=b)
            nn.init.uniform_(self.c, a=a, b=b)
    
    def forward(self, enc_output, batch_size, *args):
        if self.cell == 'GRU':
            # state shape: [num_layers, batch_size, hidden_size]
            return self.h.expand(self.h.shape[0], batch_size, self.h.shape[2])

        elif self.cell == 'LSTM':
            # h shape: [num_layers, batch_size, hidden_size]
            # c shape: [num_layers, batch_size, hidden_size]
            return (self.h.expand(self.h.shape[0], batch_size, self.h.shape[2]), self.c.expand(self.c.shape[0], batch_size, self.c.shape[2]))



# class BaseState():
#     """The base class for all initial states."""
#     def __init__(self, cell, batch_size, hidden_size, num_layers, **kwargs):
#         self.cell = cell
#         self.batch_size = batch_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.generator()

#     def generator(self):
#         print('Base')
#         raise NotImplementedError


# class ZerosState(BaseState):
#     """ """
#     def generator(self):
#         print('Zeros')
#         


# class TrainableState(BaseState):
#     """ """
#     def __init__(self, cell, batch_size, hidden_size, num_layers, a=-0.8, b=0.8, **kwargs):
#         super(self).__init__(cell, batch_size, hidden_size, num_layers, **kwargs)
#         self.a = a #min val 
#         self.b = b #max val 
        
#     def generator(self):
#         if self.cell == 'GRU':
#             # state shape: [num_layers, batch_size, hidden_size]
#             h = nn.Parameter(torch.empty(self.num_layers, self.batch_size, self.hidden_size))
#             return nn.init.uniform_(h, a=self.a, b=self.b)

#         elif self.cell == 'LSTM':
#             # h shape: [num_layers, batch_size, hidden_size]
#             h = nn.Parameter(torch.empty(self.num_layers, self.batch_size, self.hidden_size))
#             # c shape: [num_layers, batch_size, hidden_size]
#             c = nn.Parameter(torch.empty(self.num_layers, self.batch_size, self.hidden_size))
#             return (nn.init.uniform_(h, a=self.a, b=self.b), nn.init.uniform_(c, a=self.a, b=self.b))

class EncoderOutputState(BaseState):
    """ Returns an empty context."""
    def __init__(self, input_size, output_size, num_layers, aggregation="mean", **kwargs):
        super(EncoderOutputState, self).__init__(**kwargs)
        if aggregation not in ["mean", "sum", "max", "min"]:
            raise ValueError("Supported aggregations are 'mean', 'sum', 'max' or 'min'")
        self.aggregation=aggregation
        self.mapping_dims = nn.Linear(input_size, output_size)
        self.num_layers = num_layers

    def forward(self, enc_output, batch_size, *args):
        if self.aggregation == "mean":
            out = enc_output.mean(dim=-2)
        elif self.aggregation == "sum":
            out = enc_output.sum(dim=-2)
        elif self.aggregation == "max":
            out = enc_output.max(dim=-2)[0]
        elif self.aggregation == "min":
            out = enc_output.min(dim=-2)[0]
        else:
            raise ValueError(f"Aggregation '{self.aggregation}' not supported")

        # h shape: [num_layers, batch_size, hidden_size]
        out = self.mapping_dims(out)
        return out.repeat(self.num_layers, 1, 1)
        
        
    