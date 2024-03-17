# -*- coding: utf-8 -*-

import torch.nn as nn
from pytorch_metric_learning import losses, miners
class LossCAT(nn.Module):
    """
    Loss for separating category embeddings
    """
    def __init__(self, Config, gamma=4.0):
        super(LossCAT, self).__init__()
        self.gamma = gamma
        self.embDim = Config.embedDim
        self.coarse_cls_criterion = losses.LargeMarginSoftmaxLoss(Config.Ncls, self.embDim, margin=self.gamma)
        self.mine_criterion = miners.BatchEasyHardMiner(pos_strategy='hard', neg_strategy='hard')
        print("Loss for separating category embeddings")
        print("Gamma ", self.gamma)
    def forward(self, catembA, catembN, catlabel):
        dim = catlabel.shape
        catlabel = catlabel.reshape(dim[0]*dim[1]).cuda()
            
        catembA = catembA.reshape(dim[0]*dim[1], self.embDim)
        catembN = catembN.reshape(dim[0]*dim[1], self.embDim)
            
        hard_sampA = self.mine_criterion(catembA, catlabel)
        hard_sampN = self.mine_criterion(catembN, catlabel)
            
        L_CAT_A = self.coarse_cls_criterion(catembA, catlabel, hard_sampA)
        L_CAT_N = self.coarse_cls_criterion(catembN, catlabel, hard_sampN)
        L_CAT = L_CAT_A + L_CAT_N
        
        return L_CAT