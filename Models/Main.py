import torch.nn as nn
from Models.get_VT import Encoder2VT
from Models.aimNET import aimNET
from Models.Decoder import Decoder
from torch.nn.utils.rnn import pack_padded_sequence
class total_Model(nn.Module):
    def __init__(self,embed_size,vocab_size,hidden_size,iteration_times=1):
        super(total_Model, self).__init__()
        #  初始化一些模型
        self.Encoder2VT = Encoder2VT(embed_size,vocab_size,hidden_size,iteration_times=1)
        self.aimNET = aimNET(d_model=hidden_size,d_inner=2048,n_head=8,d_k=64,d_v=64,dropout=0.1)
        self.Decoder = Decoder(embed_size,vocab_size,hidden_size)
    def forward(self,images,captions,image_concepts,lengths,basic_model):
        V,T = self.Encoder2VT( images, captions, image_concepts, lengths)
        SGIR,_,_ = self.aimNET(V,T)
        # Language Modeling on word prediction
        scores = self.Decoder( SGIR, SGIR, captions, basic_model )
        
        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence( scores, lengths, batch_first=True )
        
        return packed_scores