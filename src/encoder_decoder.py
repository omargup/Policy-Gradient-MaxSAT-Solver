import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders import Encoder
from src.decoders import Decoder
from src.init_vars import BaseVar, BasicVar
from src.init_contexts import BaseContext, EmptyContext
from src.init_states import BaseState, ZerosState


class EncoderDecoder(nn.Module):
    """ 
    Parameters
    ----------
    encoder : Encoder type. If None (default), no encoder is used.
    decoder :  Decoder type.
    init_dec_var : BaseVar type. Is None (default), BasicVar is used.
    init_dec_context : BaseContext type. If None (default), EmptyContext is used.
    init_dec_state : BaseState type. If None (default), ZerosState is used.
    """

    def __init__(self, encoder=None, decoder=None, init_dec_var=None,
                 init_dec_context=None, init_dec_state=None, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.init_dec_var = init_dec_var
        self.init_dec_context = init_dec_context
        self.init_dec_state = init_dec_state
        
        if encoder is not None:
            if not issubclass(type(encoder), Encoder):
                raise TypeError("encoder must inherit from Encoder.")
        if decoder is None:
            raise TypeError("A decoder must be specified.")
        #TODO: Default RNNDecoder
        elif decoder is not None:
            if not issubclass(type(decoder), Decoder):
                raise TypeError("decoder must inherit from Decoder.")
        
        if init_dec_var is None:
            self.init_dec_var = BasicVar()
        if init_dec_context is not None:
            if not issubclass(type(init_dec_var), BaseVar):
                raise TypeError("init_dec_var must inherit from BaseVar.")

        if init_dec_context is None:
            self.init_dec_context = EmptyContext()
        if init_dec_context is not None:
            if not issubclass(type(init_dec_context), BaseContext):
                raise TypeError("init_dec_context must inherit from BaseContext.")

        if init_dec_state is None:
            self.init_dec_state = ZerosState()
        elif init_dec_state is not None:
            if not issubclass(type(init_dec_state), BaseState):
                raise TypeError("init_dec_state must inherit from BaseState.")
        
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