# -*- coding: utf-8 -*-
"""
Code for the Pose-invariant Attention Network encoder architecture 
to learn Single Object Embeddings

"""
import torch.nn as nn
import torch.nn.functional as F
import torch 
from torchvision import models
print(torch.__version__)
from einops import rearrange
from models.architecture_utils import MultiHeadAttention
        
class SingleModel(nn.Module):
    def __init__(self, inChannel, embDim, nHeads, nLayers, dropout, nCls):
        super(SingleModel, self).__init__()

        print("Using Pose-invariant Attention Network with VGG16 backbone")
        self.backbone = models.vgg16(pretrained=True)
        num_ftrs = 25088 # vgg_out (dimensionality of image features from VGG-16)
        self.nCls = nCls
        self.backbone.classifier = nn.Identity()
        self.obj_embedder = nn.Sequential(nn.Linear(num_ftrs, embDim)) 
        self.embDim = embDim
        print("Using Self-Attention: nHeads =", nHeads)
        self.obj_mha = MultiHeadAttention(nHeads, embDim, embDim, embDim, dropout)

    
    def forward(self, imgBatch):
        dim = imgBatch.shape
        b,v,c,h,w = dim
        # Rearrange multi-view image tensor to pass them all through the backbone as a batch
        # Batch (b), Views (v), Channel (c), Height (h), Width (w)--> Batch*Views, Channel, Height, Width
        imgBatch  = rearrange(imgBatch, 'b v c h w -> (b v) c h w').cuda()
        # Extract Visual Features using the VGG Backbone 
        imgFtrs = self.backbone(imgBatch) # [b*v, vgg_out]
        # Pass visual features to two FC layers 
        # Generates Single-View Object Embeddings 
        objEmbs = self.obj_embedder(imgFtrs) # [b*v, emb_dim]
        # Reorganizing input batch for self-attention layers 
        # [b*v, emb_dim] --> [b, v, emb_dim]
        objEmbs = rearrange(objEmbs, '(b v) e -> b v e',b=b) 
        # Extract Normalize Multi-View Object Embeddings 
        objContext, objAttn = self.obj_mha(objEmbs, objEmbs, objEmbs)
        MVOBJEmbs = F.normalize(torch.mean(objContext,1), dim=1, p=2)
        
        # Normalize Single-View Object Embeddings
        SVOBJEmbs = F.normalize(objEmbs, dim=2, p=2)
        
        return SVOBJEmbs, MVOBJEmbs, objAttn