import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import numpy as np
from Models.Layers import AligningLayer, IntegratingLayer, MappingLayer

__author__ = "Fenglin Liu"
# Aligning Visual Regions and Textual Concepts for Semantic-Grounded Image Representations. NeurIPS 2019. #

class aimNET(nn.Module):
    ''' Mutual Iterative Attention. '''
    
    def __init__(
           self,       
           n_head=8, d_k=64, d_v=64, d_model=512, d_inner=2048, dropout=0.1):

        super( aimNET, self ).__init__()

        assert d_model == n_head * d_k and d_k == d_v
        
        self.aligning = AligningLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.integrating = IntegratingLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.mapping = MappingLayer(d_model,d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm( d_model )

    def forward(self, V, T, return_attns=False):
        # Aligning
        V_a = self.aligning( V, T )
        T_a = self.aligning( T, V )

        # Integrating
        V_i = self.integrating(V_a)
        T_i = self.integrating(T_a)

        # Mapping            
        Refine_V = self.mapping(V_i)
        Refine_T = self.mapping(T_i)
        SGIR = self.layer_norm( Refine_T + Refine_V ) # SGIR: Semantic-Grounded Image Representations
        
        return SGIR, Refine_V, Refine_T
