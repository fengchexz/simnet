import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

from torch.autograd import Variable
import numpy as np
from Models.Layers import RefineVisualLayer, RefineTextualLayer

__author__ = "Fenglin Liu"
# Aligning Visual Regions and Textual Concepts for Semantic-Grounded Image Representations. NeurIPS 2019. #
class MappingModule(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(MappingModule, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.tanh(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
        
class MIA(nn.Module):
    ''' Mutual Iterative Attention. '''
    
    def __init__(
           self,       
           n_head=8, d_k=64, d_v=64, d_model=512, d_inner=2048, N=1, dropout=0.1):

        super( MIA, self ).__init__()

        assert d_model == n_head * d_k and d_k == d_v
        self.N = N #  iteration times
        
        self.layer_refine_V = RefineVisualLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.layer_refine_T = RefineTextualLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.mapping_module = MappingModule(d_model,d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm( d_model )

    def forward(self, V, T, return_attns=False):

        # -- Forward
        Refine_V = V
        Refine_T = T
        
        # Mutual Iterative Attention
        for i in range( self.N ):
            # Refining V
            Refine_V = self.layer_refine_V( Refine_T, Refine_V )

            # Refining T
            Refine_T = self.layer_refine_T( Refine_V, Refine_T )

            # Refining V
            Refine_V = self.layer_refine_V( Refine_V, Refine_V )

            # Refining T
            Refine_T = self.layer_refine_T( Refine_T, Refine_T )            
        Refine_T = self.mapping_module(Refine_T)
        Refine_V = self.mapping_module(Refine_V)
        SGIR = self.layer_norm( Refine_T + Refine_V ) # SGIR: Semantic-Grounded Image Representations
        
        return SGIR, Refine_V, Refine_T
