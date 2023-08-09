import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Decoder(nn.Module):
    """ """
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__()

    def forward(self, X, helper, *args):
        raise NotImplementedError


class RNNDec(Decoder):
    """ """
    def __init__(self,
                 input_size,
                 cell='GRU',
                 hidden_size=128,
                 num_layers=1,
                 dropout=0,
                 trainable_state = False,
                 output_size=1,
                 *args, **kwargs):
        super(RNNDec, self).__init__()

        self.decoder_type = cell
        self.output_size = output_size
        self.trainable_state = trainable_state
        
        # RNN
        if cell == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)
            self.init_state = None
            if trainable_state:
                self.init_state = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
                # state shape: [num_layers, batch_size=1, hidden_size]
                nn.init.xavier_normal_(self.init_state, gain=nn.init.calculate_gain('sigmoid'))
                #nn.init.uniform_(self.h, a=a, b=b)

        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
            self.init_state = None
            if trainable_state:
                self.h = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
                # h shape: [num_layers, batch_size=1, hidden_size]
                self.c = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
                # c shape: [num_layers, batch_size=1, hidden_size]
                nn.init.xavier_normal_(self.h, gain=nn.init.calculate_gain('sigmoid'))
                nn.init.xavier_normal_(self.c, gain=nn.init.calculate_gain('sigmoid'))
                #nn.init.uniform_(self.h, a=a, b=b)
                #nn.init.uniform_(self.c, a=a, b=b)
                self.init_state = (self.h, self.c)

        else:
            raise TypeError("{} is not a valid cell, try with 'LSTM' or 'GRU'.".format(self.cell))
        
        # Layer normalization
        #self.ln_input = nn.LayerNorm(input_size)
        # self.ln_output = nn.LayerNorm(hidden_size)
        # if cell == 'LSTM':
        #     self.ln_hidden = nn.LayerNorm(hidden_size)
        #     self.ln_cell = nn.LayerNorm(hidden_size)
        # else:
        #     self.ln_state = nn.LayerNorm(hidden_size)

        # Output
        self.dense_out = nn.Linear(hidden_size, output_size)

        
    def forward(self, X, state, *args):
        # X: [seq_len, batch_size, features_size=output_emb]
        # state: [num_layers, batch_size, hidden_size]
        
        #dec_input = self.ln_input(X)
        #LN Implemented for running step by step
        #TODO: full implementation of Layer Norm with LSTM
        # if self.cell == 'LSTM':
        #     h_n, c_n = state
        #     h_n = self.ln_hidden(h_n)
        #     c_n = self.ln_cell(c_n)
        #     state = (h_n, c_n)
        # else:
        #     state = self.ln_state(state)

        output, state = self.rnn(X, state)
        # output: [seq_len, batch_size, hidden_size]
        # state: [num_layers, batch_size, hidden_size]

        output = output.permute(1, 0, 2)
        # output: [batch_size, seq_len, hidden_size]
        #output = self.ln_output(output)

        logits = self.dense_out(output)
        # logits: [batch_size, seq_len, output_size={1,2}]

        return logits, state


class TransformerDec(Decoder):
    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dense_size: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 output_size: int=1,
                 *args, **kwargs):
        super(TransformerDec, self).__init__()
        self.decoder_type = "Transformer"
        self.output_size = output_size

        self.d_model = d_model

        # Custom encoder
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, dense_size, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        
        # Output
        self.dense_out = nn.Linear(d_model, output_size)


    def generate_square_subsequent_mask(self, sz: int)  -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        

    def forward(self, X, mask, *args):
        # X: [seq_len, batch_size, features_size=output_emb]
        # mask: [seq_len, seq_len]

        output = self.transformer_encoder(X, mask)
        # output: [seq_len, batch_size, d_model]
        
        output = output.permute(1, 0, 2)
        # output: [batch_size, seq_len, d_model]
        
        logits = self.dense_out(output)
        # logits: [batch_size, seq_len, output_size]  # output_size is 1 or 2.

        return logits