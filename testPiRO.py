#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Code for evaluating trained models on single and multi-view 
pose-invariant classification and retrieval tasks  
===============================================================================
"""
"""
Load Libraries
"""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision.datasets as dset
import torch
import sys 
print(torch.__version__)
from utils.InferenceUtility_large import evaluate_performance_single, evaluate_performance_dual
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Load utility functions 
-------------------------------------------------------------------------------
SingleModel, DualModel: Pose-invariant Attention Network Architecture with 
                        VGG Backbone in the Single and Dual Embedding Space
InferenceUtility_large: Functions for inference and computation of PICR tasks using 
                  single and dual embeddings
helperFunctions: Other utility functions
ConfigLearn: Training and Testing Configurations for different datasets and the 
             hyper-parameters for the corresponding datasets
-------------------------------------------------------------------------------
"""


from ConfigLearn import ConfigOOWL, ConfigMNet40, ConfigFG3D, HyperParams

"""
Input information and hyper-parameters from user
"""
dataset = sys.argv[1] # OOWL, MNet40, FG3D
emb_space = sys.argv[2] # single or dual
model_path = sys.argv[3] # model path

hp = HyperParams(dataset)

print("Large Margin Softmax Loss for classification: ", hp.gamma, " nHeads: ", hp.nHeads, "nLayers: ", hp.nLayers )
"""
Load configuration files for the dataset
"""
# Datasets for testing PICR  
if dataset == 'OOWL':
    Config = ConfigOOWL(hp.case, hp.embDim, hp.batchSize, hp.alpha, hp.n_randsamp_class, hp.seed_inp)
elif dataset == 'MNet40':
    Config = ConfigMNet40(hp.case, hp.embDim, hp.batchSize, hp.alpha, hp.n_randsamp_class, hp.seed_inp)
elif dataset == 'FG3D':
    Config = ConfigFG3D(hp.case, hp.embDim, hp.batchSize, hp.alpha, hp.n_randsamp_class, hp.seed_inp)
else:
    print("Wrong Dataset")

print("Loaded data")

"""
    Main script for evaluation
"""
if emb_space == 'single':
    from models.VGG_PAN_SingleEmb import SingleModel
    trained_model = SingleModel(Config.inpChannel, Config.embedDim, hp.nHeads, hp.nLayers, hp.dropout, Config.Ncls).to(device)
    print("Evaluating model trained in the single embedding space")
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()
    orm, crm, clea, svoc = evaluate_performance_single(dataset, Config, trained_model, 1)
    mvcrm, mvclea, mvoc, mvoret = evaluate_performance_single(dataset, Config, trained_model, Config.N_G)
elif emb_space == 'dual':
    from models.VGG_PAN_DualEmb import DualModel
    trained_model = DualModel(Config.inpChannel, Config.embedDim, hp.nHeads, hp.nLayers, hp.dropout, Config.Ncls).to(device)
    print("Evaluating model trained in the dual embedding space")
    trained_model.load_state_dict(torch.load(model_path))    
    trained_model.eval()
    orm, crm, clea, svoc = evaluate_performance_dual(dataset, Config, trained_model, 1)
    mvcrm, mvclea, mvoc, mvoret = evaluate_performance_dual(dataset, Config, trained_model, Config.N_G)
else: 
    print("Wong embedding space. enter single or dual")

print("Loaded model weights")

print("-------------------------------------------------------------------")    
print("Test Results: \n")   
print("-------------------------- Single-View ----------------------------")
print("SV Classification Accuracy: Category {} % | Object {} %|\n".format(clea, svoc))  
print("SV  Retrieval mAP: Category {} %| Object {} %|\n".format(crm, orm))
print("-------------------------- Multi-View ----------------------------")    
print("MV Classification Accuracy: Category {} %| Object {} %| ".format(mvclea, mvoc))  
print("MV Retrieval mAP: Category {} %| Object {} %| ".format(mvcrm, mvoret))
print("-------------------------------------------------------------------") 