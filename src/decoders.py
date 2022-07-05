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




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

        

class TransformerDecoder(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 trg_vocab_size: int,
                 d_model: int,
                 num_heads: int,
                 num_enc_layers: int,
                 num_dec_layers: int,
                 ffn_size: int,
                 dropout: float,
                 activation: str):
        
        super().__init__()
        
        self.d_model = d_model

        #Embeddings
        self.enc_embedding = nn.Embedding(num_embeddings = src_vocab_size, 
                                          embedding_dim = d_model)
        self.dec_embedding = nn.Embedding(num_embeddings = trg_vocab_size,
                                          embedding_dim = d_model)

        
        #Positional encoder
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        #Custom encoder
        #encoder_layer = TransformerEncoderLayer(d_model, num_heads, ffn_size, dropout, activation)
        #encoder_norm = nn.LayerNorm(d_model)
        #self.encoder = TransformerEncoder(encoder_layer, num_enc_layers, encoder_norm)
        
        #Custom decoder
        #decoder_layer = TransformerDecoderLayer(d_model, num_heads, ffn_size, dropout, activation)
        #decoder_norm = nn.LayerNorm(d_model)
        #self.decoder = TransformerDecoder(decoder_layer, num_dec_layers, decoder_norm)
        
        #Transformer
        self.transformer = nn.Transformer(d_model = d_model, 
                                          nhead = num_heads,
                                          num_encoder_layers = num_enc_layers,
                                          num_decoder_layers = num_dec_layers,
                                          dim_feedforward = ffn_size,
                                          dropout = dropout,
                                          activation = activation,
                                          custom_encoder = None, #self.encoder,
                                          custom_decoder = None) #self.decoder)
        
        #Linear layer
        self.fc = nn.Linear(d_model, trg_vocab_size)
        
        #Initialize weights
        self.init_weights()
    

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)
        
        nn.init.normal_(self.enc_embedding.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.dec_embedding.weight, mean=0.0, std=1.0)

        
    def forward(self, src: torch.tensor, trg: torch.tensor):
        trg_mask = self.generate_square_subsequent_mask(len(trg)).to(device)

        src = self.enc_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        trg = self.dec_embedding(trg) * math.sqrt(self.d_model)
        trg = self.pos_encoder(trg)

        output = self.transformer(src, trg, tgt_mask = trg_mask)
        #>>output: [trg_len, batch_size, d_model]
        output = self.fc(output)
        #>>output: [trg_len, batch_size, trg_vocab_size]

        return output