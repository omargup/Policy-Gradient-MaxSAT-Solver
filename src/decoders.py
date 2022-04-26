import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings import BaseEmbedding

class Decoder(nn.Module):
    """ """
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def forward(self, X, state):
        raise NotImplementedError


class RNNDecoder(Decoder):
    """ """
    def __init__(self, input_size, cell='GRU', assignment_emb=None, variable_emb=None, context_emb=None,
                    hidden_size=128, num_layers=1, dropout=0, clip_logits_c=0, **kwargs):
        super(RNNDecoder, self).__init__(**kwargs)
        self.cell = cell
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.c = clip_logits_c
        #TODO: Check if clip_logits_c is intenger >= 0 
        
        # Embeddings
        self.assignment_embedding = assignment_emb
        if assignment_emb is None:
            raise TypeError("An assignment_emb must be specified.")
        if assignment_emb is not None:
            if not issubclass(type(assignment_emb), BaseEmbedding):
                raise TypeError("assignment_emb must inherit from BaseEmbedding.")

        self.variable_embedding = variable_emb
        if variable_emb is None:
            raise TypeError("A variable_emb must be specified.")
        if variable_emb is not None:
            if not issubclass(type(variable_emb), BaseEmbedding):
                raise TypeError("variable_emb must inherit from BaseEmbedding.")
        
        self.context_embedding = context_emb
        if context_emb is None:
            raise TypeError("A context_emb must be specified.")
        if context_emb is not None:
            if not issubclass(type(context_emb), BaseEmbedding):
                raise TypeError("context_emb must inherit from BaseEmbedding.")
        
        # RNN
        if self.cell == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)
        elif self.cell == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        else:
            raise TypeError("{} is not a valid cell, try with 'LSTM' or 'GRU'.".format(self.cell))
        # TODO: validate input size == variable_emb + assig_emb + context
        # TODO: Projection or nn.Linear?

        # Output
        self.dense_out = nn.Linear(hidden_size, 2)

    def forward(self, X, state):
        var, a_prev, context = X
        # ::var:: [batch_size, seq_len, feature_size]
        #       ex: [[[0], [1], [2]], [[0], [1], [2]]]
        # ::a_prev:: [batch_size, seq_len, feaure_size=1]
        #       ex.: [[[0], [0], [1]], [[1], [0], [1]]]
        # ::context:: [batch_size, feature_size]
        # ::state:: [num_layers, batch_size, hidden_size]
        var = self.variable_embedding(var).permute(1, 0, 2)
        # ::var:: [seq_len, batch_size, features_size=var_embedding_size]
        a_prev = self.assignment_embedding(a_prev).permute(1, 0, 2)
        # ::a_prev:: [seq_len, batch_size, feature_size=assig_embedding_size]
        context = self.context_embedding(context)
        # ::context:: [batch_size, features_size=context_embedding_size]
        
        # Broadcasting context
        context = context.repeat(var.shape[0], 1, 1)
        # ::context:: [seq_len, batch_size, features_size]
        dec_input = torch.cat((var, a_prev, context), -1)
        # ::dec_input:: [seq_len, batch_size, features_size=input_size]

        output, state = self.rnn(dec_input, state)
        # ::output:: [seq_len, batch_size, hidden_size]
        # ::state:: [num_layers, batch_size, hidden_size]

        output = output.permute(1, 0, 2)
        # ::output:: [batch_size, seq_len, hidden_size]

        logits = self.dense_out(output)
        # ::logits:: [batch_size, seq_len, 2]

        # clipped logits
        #TODO: Check c * tanh(logits)
        if self.c > 0:
            logits = self.c * F.tanh(logits)

        return logits, state