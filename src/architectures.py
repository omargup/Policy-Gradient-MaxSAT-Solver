import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings import Embedding, BasicEmbedding
from src.baselines import BaselineRollout
from src.init_states import BaseState, ZerosState, TrainableState
from src.init_context import BaseContext, EmptyContext


import src.utils as utils


class Encoder(nn.Module):
    """ """
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """ """
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    #def init_state(self, enc_outputs, *args):
    #    raise NotImplementedError

    def forward(self, X, state):
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
            if not issubclass(type(self.embedding), Embedding):
                raise TypeError("embedding must inherit from Embedding.")
        
        # RNN
        if cell == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout = dropout)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout)
        else:
            raise TypeError("{} is not a valid cell, try with 'LSTM' or 'GRU'.".format(self.cell))
        

    def forward(self, formula, *args):
        # formula shape: [batch_size, seq_len]
        X = self.embedding(formula)
        # X shape: [batch_size, seq_len, embedding_size]
        X = X.permute(1, 0, 2)
        # X shape: [seq_len, batch_size, embedding_size]
        output, state = self.rnn(X)  # Initial state is zeros
        # output shape: [seq_len, batch_size, hidden_size]
        # state shape: [num_layers, batch_size, hidden_size]
        return output, state


class RNNDecoder(Decoder):
    """ """
    def __init__(self, input_size, cell='GRU', assignment_emb=None, variable_emb=None,
                    hidden_size=128, num_layers=1, dropout=0, clip_logits_c=0, **kwargs):
        super(RNNDecoder, self).__init__(**kwargs)
        self.cell = cell
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.c = clip_logits_c
        
        # Embeddings
        self.assignment_embedding = assignment_emb
        if assignment_emb is None:
            raise TypeError("An assignment_emb must be specified.")
        if assignment_emb is not None:
            if not issubclass(type(assignment_emb), Embedding):
                raise TypeError("assignment_emb must inherit from Embedding.")

        self.variable_embedding = variable_emb
        if variable_emb is None:
            raise TypeError("A variable_emb must be specified.")
        if variable_emb is not None:
            if not issubclass(type(variable_emb), Embedding):
                raise TypeError("variable_emb must inherit from Embedding.")
        
        # RNN
        if self.cell == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)
        elif self.cell == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        else:
            raise TypeError("{} is not a valid cell, try with 'LSTM' or 'GRU'.".format(self.cell))
        # TODO: validate input size == variable_emb + assig_emb + context

        # Output
        self.dense_out = nn.Linear(hidden_size, 2)


    def forward(self, X, state):
        var, a_prev, context = X
        # var shape: [batch_size, seq_len, feature_size]
        #       ex: [[[-2], [1], [-3]], [[-2], [1], [-3]]]
        # a_prev shape: [batch_size, seq_len, feaure_size=1]
        #       ex.: [[[0], [0], [1]], [[1], [0], [1]]]
        # context shape: [batch_size, feature_size]
        # state: [num_layers, batch_size, hidden_size]
        
        a_prev = self.assignment_embedding(a_prev).permute(1, 0, 2)
        # a_prev shape: [seq_len, batch_size, feature_size=assig_embedding_size]
        var = self.variable_embedding(var).permute(1, 0, 2)
        # var shape: [seq_len, batch_size, features_size=var_embedding_size]
        # Broadcasting context
        context = context.repeat(var.shape[0], 1, 1)
        # context shape: [seq_len, batch_size, features_size]

        dec_input = torch.cat((var, a_prev, context), -1)
        # dec_input shape: [seq_len, batch_size, features_size=input_size]

        output, state = self.rnn(dec_input, state)
        # output shape: [seq_len, batch_size, hidden_size]
        # state shape: [num_layers, batch_size, hidden_size]

        output = output.permute(1, 0, 2)
        # output shape: [batch_size, seq_len, hidden_size]

        logits = self.dense_out(output)
        # logits: [batch_size, seq_len, 2]

        # clipped logits
        if self.c > 0:
            logits = self.c * F.tanh(logits)

        return logits, state


class EncoderDecoder(nn.Module):
    """ 
    Parameters
    ----------
    encoder : Encoder type. If None (default), no encoder is used.
    decoder :  Decoder type.
    init_dec_state :  BaseState type. If None (default), ZerosState is used.
    init_dec_context : BaseContext type. If None (default), EmptyContext is used."""

    def __init__(self, encoder=None, decoder=None, init_dec_state=None, init_dec_context=None, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.init_dec_state = init_dec_state
        self.init_dec_context = init_dec_context

        if encoder is not None:
            if not issubclass(type(encoder), Encoder):
                raise TypeError("encoder must inherit from Encoder.")
        if decoder is None:
            raise TypeError("A decoder must be specified.")
        #TODO: Default RNNDecoder
        elif decoder is not None:
            if not issubclass(type(decoder), Decoder):
                raise TypeError("decoder must inherit from Decoder.")
        
        if init_dec_state is None:
            self.init_dec_state = ZerosState()
        elif init_dec_state is not None:
            if not issubclass(type(init_dec_state), BaseState):
                raise TypeError("init_dec_state must inherit from BaseState.")
        
        if init_dec_context is None:
            self.init_dec_context = EmptyContext()
        if init_dec_context is not None:
            if not issubclass(type(init_dec_context), BaseContext):
                raise TypeError("context must inherit from BaseContext.")
        

    def forward(self, dec_input, enc_input=None, *args):
        var, a_prev = (dec_input)
        # var: [batch_size, seq_len]
        # a_prev: [batch_size, seq_len]

        # Encoder
        enc_outputs = None
        if self.encoder is not None:
            if enc_input is None:
                raise TypeError("enc_input must be specified.")
            enc_outputs = self.encoder(enc_input, *args)
        
        # Decoder State
        dec_state = self.init_dec_state(enc_outputs, *args)
        
        # Decoder Context
        if self.context is None:
            # dec_context must be: [batch_size, feature_size]
            dec_context = torch.rand([var.shape[0], 0]) #empty context
        if self.context is not None:
            dec_context = self.context(enc_outputs, *args)
        
        dec_input = (var, a_prev, dec_context)    
        logits, state= self.decoder(dec_input, dec_state)

        return logits, state
