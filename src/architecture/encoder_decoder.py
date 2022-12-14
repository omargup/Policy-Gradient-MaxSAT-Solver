import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architecture.encoders import Encoder
from src.architecture.decoders import Decoder
from src.initializers.var_initializer import BaseVar, BasicVar
from src.initializers.context_initializer import BaseContext, EmptyContext
from src.initializers.state_initializer import BaseState, ZerosState


class EncoderDecoder(nn.Module):
    """ The encoder-decoder architecture.
    
    Parameters
    ----------
    - encoder (Encoder type): The encoder. If None (default), no encoder is used.
    - decoder (Decoder type): The decoder.
    - dec_var_initializer (BaseVar type): Variable initializer for the decore. If None (default), BasicVar is used.
    - dec_context_initializer (BaseContext type): Context initializer for the decoder. If None (default), EmptyContext is used.
    - dec_state_initializer (BaseState type): State initializer for the decoder. If None (default), ZerosState is used.
    """

    def __init__(self,
                 encoder=None,
                 decoder=None,
                 dec_var_initializer=None,
                 dec_context_initializer=None,
                 dec_state_initializer=None,
                 **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.dec_var_initializer = dec_var_initializer
        self.dec_context_initializer = dec_context_initializer
        self.dec_state_initializer = dec_state_initializer
        
        if encoder is not None:
            if not issubclass(type(encoder), Encoder):
                raise TypeError("encoder must inherit from Encoder.")
        
        if decoder is None:
            raise TypeError("A decoder must be specified.")
        #TODO: Default RNNDecoder
        elif decoder is not None:
            if not issubclass(type(decoder), Decoder):
                raise TypeError("decoder must inherit from Decoder.")
        
        if dec_var_initializer is None:
            self.dec_var_initializer = BasicVar()
        if dec_var_initializer is not None:
            if not issubclass(type(dec_var_initializer), BaseVar):
                raise TypeError("dec_var_initializer must inherit from BaseVar.")

        if dec_context_initializer is None:
            self.dec_context_initializer = EmptyContext()
        if dec_context_initializer is not None:
            if not issubclass(type(dec_context_initializer), BaseContext):
                raise TypeError("dec_context_initializer must inherit from BaseContext.")

        if dec_state_initializer is None:
            self.dec_state_initializer = ZerosState()
            #self.init_dec_state = ZerosState(cell, hidden_size, num_layers)
        if dec_state_initializer is not None:
            if not issubclass(type(dec_state_initializer), BaseState):
                raise TypeError("dec_state_initializer must inherit from BaseState.")
        
    def forward(self, dec_input, enc_input=None, *args):
        raise NotImplementedError

        # var, a_prev = (dec_input)
        # # var: [batch_size, seq_len]
        # # a_prev: [batch_size, seq_len]

        # # Encoder
        # enc_outputs = None
        # if self.encoder is not None:
        #     if enc_input is None:
        #         raise TypeError("enc_input must be specified.")
        #     enc_outputs = self.encoder(enc_input, *args)
        
        # # Decoder State
        # dec_state = self.init_dec_state(enc_outputs, *args)
        
        # # Decoder Context
        # if self.context is None:
        #     # dec_context must be: [batch_size, feature_size]
        #     dec_context = torch.rand([var.shape[0], 0]) #empty context
        # if self.context is not None:
        #     dec_context = self.context(enc_outputs, *args)
        
        # dec_input = (var, a_prev, dec_context)    
        # logits, state= self.decoder(dec_input, dec_state)

        # return logits, state