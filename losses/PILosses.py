# -*- coding: utf-8 -*-
"""
Code for the Pose-invariant losses in the Dual Embedding space

"""
import torch
import torch.nn.functional as F   
import torch.nn as nn
from utils.helperFunctions import expanded_pairwise_distances
    
class PILossOBJ(nn.Module):
    """
    Pose-invariant Loss for the Object Embedding Space
    """
    def __init__(self, alpha=0.4, beta = 2.0, lamda=2.0):
        super(PILossOBJ, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.lamda = lamda
        print("PI loss for object embedding space")
        print("alpha ", alpha, " beta ", beta, "lamda ", lamda)
    def forward(self, SV_A, SV_N, MV_A, MV_N, size_average=True):
        """ Find confusing instances from a pair of objects from same category """
        dSV_A_SVN_N = expanded_pairwise_distances(SV_A,SV_N)
        nA, nN, nB = dSV_A_SVN_N.shape
        confuser_A_N = torch.argmin(dSV_A_SVN_N.reshape(nA*nN, nB),0)
        confuser_A = confuser_A_N/nN
        confuser_N = confuser_A_N%nN
        f_confuser_A = torch.stack([SV_A[int(confuser_A[i]), i, :] for i in range(nB)], 0)
        f_confuser_N = torch.stack([SV_N[int(confuser_N[i]), i, :] for i in range(nB)], 0)
        """ Compute inter-class distance between MV embeddings of both objects """
        MV_AMV_N_inter = torch.cdist(MV_A.unsqueeze(1),MV_N.unsqueeze(1)).squeeze()
        """ Compute inter-class distance between confuser embeddings of both objects """
        MC_inter = torch.min(torch.min(dSV_A_SVN_N,0).values,0).values
        """ Compute intra-class distance between MV and confuser embeddings 
            for each object """
        A_clstr = torch.cdist(MV_A.unsqueeze(1),f_confuser_A.unsqueeze(1)).squeeze()
        C_intra = torch.cdist(MV_N.unsqueeze(1),f_confuser_N.unsqueeze(1)).squeeze()
        """ Clustering loss component """
        loss_M_cluster = F.relu(A_clstr - self.alpha)
        loss_C_cluster = F.relu(C_intra - self.alpha)
        """ Separation loss component """
        loss_MC_inter = F.relu(self.beta - MC_inter)
        loss_MV_AMV_N_inter = F.relu(self.beta - MV_AMV_N_inter)
        """ Total loss """
        losses = self.lamda*(loss_MC_inter+loss_MV_AMV_N_inter)+loss_M_cluster+loss_C_cluster
        #print("PI OBJ LOSS",losses, loss_MC_inter, loss_MV_AMV_N_inter, loss_M_cluster, loss_C_cluster)
        info_quads = [torch.numel(torch.nonzero(losses)), MC_inter.pow(0.5).mean(), 0.5*(A_clstr.pow(0.5).mean()+C_intra.pow(0.5).mean()), torch.min(MC_inter.pow(0.5))]
        avg_loss =  losses.mean() if size_average else losses.sum()
        return avg_loss, info_quads
    

    
class PILossCAT(nn.Module):
    """
    Pose-invariant Loss for the Category Embedding Space
    """
    def __init__(self, theta=0.4, lamda=2.0):
        super(PILossCAT, self).__init__()
        self.theta = theta
        self.lamda = lamda
        print("PI loss for category embedding space")
        print("theta ", theta, "lamda ", lamda)
    def forward(self, SV_A, SV_N, MV_A, MV_N, size_average=True):
        """ Compute distance between SV and MV embeddings of same object"""
        d_SV_A_MV_A = torch.mean(torch.cdist(SV_A, MV_A.unsqueeze(1)).squeeze(),1)
        d_SV_N_MV_N = torch.mean(torch.cdist(SV_N, MV_N.unsqueeze(1)).squeeze(),1)
        """ Compute distance between MV embeddings of different objects from the 
            same category """
        d_MV_A_MV_N = torch.cdist(MV_A.unsqueeze(1),MV_N.unsqueeze(1)).squeeze()
        """ Total loss """
        loss_SV_A_MV_A = F.relu(d_SV_A_MV_A - self.theta)
        loss_SV_N_MV_N = F.relu(d_SV_N_MV_N - self.theta)
        loss_MV_A_MV_N = F.relu(d_MV_A_MV_N - self.theta)
        losses = loss_SV_A_MV_A + loss_SV_N_MV_N + loss_MV_A_MV_N
        #print("PI_CAT_LOSS",losses, loss_SV_A_MV_A, loss_SV_N_MV_N, loss_MV_A_MV_N)
        avg_loss =  losses.mean() if size_average else losses.sum()
        return avg_loss
    
