import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


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

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """ """
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class Embedding(nn.Module):
    """ """
    def __init__(self, **kwargs):
        super(Embedding, self).__init__(**kwargs)
    
    def forward(self, X, *args):
        raise NotImplementedError

class EmbeddingForLiterals(Embedding):
    """ """
    def __init__(self, num_labels, embedding_size, **kwargs):
        super(EmbeddingForLiterals, self).__init__(**kwargs)
        self.num_labels = num_labels
        self.embedding = nn.Linear(num_labels, embedding_size)

    def forward(self, X):
        #X: [batch_size, seq_len=1]
        if X < 0:
            X = X+n
        elif X > 0:
            X = X+n-1
    
        X = F.one_hot(X, self.num_labels).float()
        #x: [batch_size, seq_len=1, num_features=num_labels]
        return self.embedding(X)
        #x: [batch_size, seq_len=1, num_features=embedding_size]
        #[-n, ..., -1, 1, .., n]

class EmbeddingOHLinear(Embedding):
    """ """
    def __init__(self, num_labels, embedding_size, **kwargs):
        super(EmbeddingOHLinear, self).__init__(**kwargs)
        self.num_labels = num_labels
        self.embedding = nn.Linear(num_labels, embedding_size)

    def forward(self, X):
        #X: [batch_size, seq_len=1]
        X = F.one_hot(X, self.num_labels).float()
        #x: [batch_size, seq_len=1, num_features=num_labels]
        return self.embedding(X)
        #x: [batch_size, seq_len=1, num_features=embedding_size]



class BasicRNN(Decoder):
    """ """
    def __init__(self, cell, input_size, embedding_size, hidden_size, output_size,
                 num_layers, dropout=0, **kwargs):
        super(BasicRNN, self).__init__(**kwargs)
        self.cell = cell
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.var_embedding = EmbeddingOneHotW(num_labels = input_size,
                                       embedding_size = embedding_size)
        self.assig_embedding = EmbeddingOneHotW(num_labels = 3,
                                         embedding_size = embedding_size)
        if self.cell == 'GRU':
            self.rnn = nn.GRU(embedding_size*2, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        elif self.cell == 'LSTM':
            self.rnn = nn.LSTM(embedding_size*2, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        else:
            raise TypeError("{} is not a valid cell, try with 'LSTM' or 'GRU'.".format(self.cell))


        self.dense_out = nn.Linear(hidden_size, output_size)

    def init_state_basicrnn(self, *args):
        #state: [num_layers, batch_size, hidden_size]
        return torch.zeros(self.num_layers, 1, self.hidden_size)

    def forward(self, X, state):
        X, a_prev = X
        #X: [batch_size, seq_len=1]
        #       i.e.: X=[[-2]] stands for not x_2
        #a_prev: [batch_size, seq_len=1]
        #       i.e.: a_prev=[[0]] means thar the prev. assignment was 0
        #state: [num_layers, batch_size, hidden_size]

        X = self.var_embedding(X)
        #X: [batch_size, seq_len=1, num_features=embedding_size]
        a_prev = self.assig_embedding(a_prev)
        #a_prev: [batch_size, seq_len=1, num_features=embedding_size]
        
        input_t = torch.cat((X, a_prev), -1)
        output, state = self.rnn(input_t, state)
        #output: [batch_size, seq_len=1, hidden_size]
        #state: [num_layers, batch_size, hidden_size]
        logits = self.dense_out(output)
        #output: [batch_size, seq_len=1, output_size]

        #if clipped_logits:
        #    logits = self.c * F.tanh(logits)

        return logits, state
    



class BaseAggregator(nn.Module):
    def __init__(self, output_size):
        super(BaseAggregator, self).__init__()
        self.output_size = output_size

    def forward(self, src):
        raise NotImplementedError("This feature hasn't been implemented yet!")

class ConcatenateAggregator(BaseAggregator):
    def forward(self, src):
        return torch.flatten(src, start_dim=1)

class SumAggregator(BaseAggregator):
    def forward(self, src):
        return torch.sum(src, dim=1, keepdim=False)



class Baseline(nn.Module):
    """  """
    def __init__(self, **kwargs):
        super(Baseline, self).__init__(**kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class BaselineRollout(Baseline):
    """ """
    def __init__(self, num_rollouts, sampled, **kwargs):
        super(BaselineRollout, self).__init__(**kwargs)
        self.num_rollouts = num_rollouts
        self.sampled = sampled

    def forward(self, formula, logits):
        #logits: [seq_len, feature_size=2]

        #if greedy rollout
        if not self.sampled: 
            assignment = torch.argmax(logits, dim=-1)
            #assigment: [seq_len]
            _, num_sat, _ = utils.assignment_verifier(formula,
                                                      assignment=assignment.tolist())
            baseline = torch.tensor(num_sat, dtype=float)
        
        #if sampled rollouts
        else:
            action_softmax = F.softmax(logits, dim = -1)
            action_dist = torch.distributions.Categorical(action_softmax)
            #action_dist: [seq_len, feature_size=2]
            
            num_sats = []
            for i in range(self.num_rollouts):
                assigment = action_dist.sample()
                #assigment: [seq_len]

                _, num_sat, _ = utils.assignment_verifier(formula,
                                                          assignment=assigment.tolist())
                num_sats.append(num_sat)
        
            baseline = torch.tensor(num_sats, dtype=float).mean()

        return baseline
