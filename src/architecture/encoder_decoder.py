import torch
import torch.nn as nn

from src.architecture.decoders import Decoder
from src.initializers.var_initializer import BaseVar, BasicVar
from src.initializers.context_initializer import BaseContext, EmptyContext
from src.architecture.embeddings import BaseEmbedding


class PolicyNetwork(nn.Module):
    """ The policy architecture.
    
    Parameters
    ----------
    - emb_module (Embedding type). The embedding module.
    - decoder (Decoder type): The decoder.
    - dec_var_initializer (BaseVar type): Variable initializer for the decore. If None (default), BasicVar is used.
    - dec_context_initializer (BaseContext type): Context initializer for the decoder. If None (default), EmptyContext is used.
    """

    def __init__(self,
                 formula,
                 num_variables,
                 n2v_emb = None,
                 emb_module=None,
                 decoder=None,
                 dec_var_initializer=None,
                 dec_context_initializer=None,
                 *args, **kwargs):
        super(PolicyNetwork, self).__init__()
        self.emb_module = emb_module
        self.decoder = decoder

        if emb_module is None:
            raise TypeError("An emb_module must be specified.")
        if emb_module is not None:
            if not issubclass(type(emb_module), BaseEmbedding):
                raise TypeError("emb_module must inherit from BaseEmbedding.")
        
        if decoder is None:
            raise TypeError("A decoder must be specified.")
        elif decoder is not None:
            if not issubclass(type(decoder), Decoder):
                raise TypeError("decoder must inherit from Decoder.")
        
        if dec_var_initializer is None:
            dec_var_initializer = BasicVar()
        if dec_var_initializer is not None:
            if not issubclass(type(dec_var_initializer), BaseVar):
                raise TypeError("dec_var_initializer must inherit from BaseVar.")

        if dec_context_initializer is None:
            dec_context_initializer = EmptyContext()
        if dec_context_initializer is not None:
            if not issubclass(type(dec_context_initializer), BaseContext):
                raise TypeError("dec_context_initializer must inherit from BaseContext.")

        # Initialize Decoder Variables 
        self.dec_vars = dec_var_initializer(n2v_emb, formula, num_variables)
        # dec_vars: [batch_size=1, seq_len=num_vars, feature_size={num_vars, 2*n2v_emb_dim}]

        # Initialize Decoder Context
        self.dec_context = dec_context_initializer(n2v_emb, formula, num_variables)
        # dec_context: [batch_size=1, feature_size={0, 2*n2v_emb_dim}]

        # Initialize action_prev at time t=0 with token 2 (sos).
        self.init_action = torch.tensor([2], dtype=torch.long).reshape(-1,1,1)
        # init_action: [batch_size=1, seq_len=1, feature_size=1]

        # Initialize Decoder state
        if (self.decoder.decoder_type == "GRU") or (self.decoder.decoder_type == "LSTM"):
            self.dec_init_state =  decoder.init_state
            # dec_init_state: [num_layers, batch_size=1, hidden_size]

        
    def forward(self, dec_input, helper, *args):
        variable, assignment, context = dec_input
        # variable: [batch_size, seq_len, features_size], feature_size could be n or 2*n2v_emb_size
        # assignment: [batch_size, seq_len, features_size=3]
        # context: [batch_size, seq_len, features_size], feature_size could be 0 or 2*n2v_emb_size.
        
        # helper is the state if RNN or mask if Transformer
        # dec_state: [num_layers, batch_size, hidden_size] if decoder is GRU or LSTM.

        emb_vec = self.emb_module(variable, assignment, context)
        # emb_vec: [batch_size, seq_len, features_size=output_emb]
        emb_vec = emb_vec.permute(1, 0, 2)
        # emb: [seq_len, batch_size, features_size=output_emb]

        if (self.decoder.decoder_type == "GRU") or (self.decoder.decoder_type == "LSTM"):
            logits, dec_state = self.decoder(emb_vec, helper)  # helper is the state
            # logits: [batch_size, seq_len, output_size={1,2}]
            # dec_state: [num_layers, batch_size, hidden_size]
            
            return logits, dec_state
        
        else:  # Transformer
            logits = self.decoder(emb_vec, helper)  # helper is the mask
            # logits: [batch_size, seq_len, output_size={1,2}]
            
            return logits



