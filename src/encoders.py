import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings import BaseEmbedding

class Encoder(nn.Module):
    """ """
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class RNNEncoder(Encoder):
    """ """
    def __init__(self, cell='GRU', embedding=None, embedding_size=128, hidden_size=128,
                    num_layers=1, dropout=0, **kwargs):
        super(RNNEncoder, self).__init__(**kwargs)

        # Embedding
        self.embedding = embedding
        if embedding is None:
            raise TypeError("An embedding must be specified.")
        elif embedding is not None:
            if not issubclass(type(self.embedding), BaseEmbedding):
                raise TypeError("embedding must inherit from BaseEmbedding.")
        
        # RNN
        if cell == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout = dropout)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout)
        else:
            raise TypeError("{} is not a valid cell, try with 'LSTM' or 'GRU'.".format(self.cell))

    def forward(self, formula, *args):
        # ::formula:: [batch_size, seq_len, features_size]
        X = self.embedding(formula)
        # ::X:: [batch_size, seq_len, embedding_size]
        X = X.permute(1, 0, 2)
        # ::X:: [seq_len, batch_size, embedding_size]
        output, state = self.rnn(X)  # Initial state is zeros
        # ::output:: [seq_len, batch_size, hidden_size]
        # ::state:: [num_layers, batch_size, hidden_size]
        return output, state