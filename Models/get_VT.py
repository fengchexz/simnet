import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from Models.Decoder import Decoder
from torch.autograd import Variable
import numpy as np

# 用预训练ResNet152获取图像的视觉特征
class EncoderCNN(nn.Module):
    def __init__(self,hidden_size):
        super(EncoderCNN,self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]
        resnet_conv = nn.Sequential(*modules)

        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d(7)
        self.affine_a = nn.Linear(2048,hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()
    def init_weights(self):
        init.kaiming_uniform(self.affine_a.weight,mode='fan_in')
        self.affine_a.bias.data.fill_(0)
    def forward(self,images):
        A = self.resnet_conv(images)

        a_g = self.avgpool(A)
        a_g = a_g.view(a_g.size(0),-1)

        V = A.view(A.size(0),A.size(1),-1).transpose(1,2)
        V = F.relu(self.affine_a(self.dropout(V)))
        return V
class Encoder2VT(nn.Module):
    def __init__(self,embed_size,vocab_size,hidden_size):
        super(Encoder2VT, self).__init__()
        self.encoder_image = EncoderCNN(hidden_size)
        self.encoder_concept = nn.Embedding(vocab_size,embed_size)
        self.decoder = Decoder( embed_size, vocab_size, hidden_size )
        self.encoder_concept.weight = self.decoder.caption_embed.weight
        assert embed_size == hidden_size
    def forward(self,images,captions,image_concepts,lengths):
        # V=[ v_1, ..., v_k ] 
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_image_parallel = torch.nn.DataParallel( self.encoder_image, device_ids=device_ids )
            V = encoder_image_parallel( images ) 
        else:
            V = self.encoder_image( images )
        # T=[ t_1, ..., t_k ] 
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_concept_parallel = torch.nn.DataParallel( self.encoder_concept, device_ids=device_ids )
            T = encoder_concept_parallel( image_concepts ) 
        else:
            T = self.encoder_concept( image_concepts )
        return V,T