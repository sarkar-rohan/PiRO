#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for inference on Pose-invariant Classification and Retrieval (PICR) 
category and object-level tasks 
"""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from tqdm import tqdm 
import torch
import numpy as np
import faiss
from functools import reduce
import operator
from utils.rank_metrics import calculate_mAP_large
from utils.DataUtility_PiRO import loadDataset
import time

"""
===============================================================================
    Class for Fast Similarity Search for classification and retrieval 
    based on k-Nearest Neighbors. This is based on Faiss library 
    Reference: https://github.com/facebookresearch/faiss
===============================================================================
"""
class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None # indexing structure
        self.y = None # labels
        self.k = k # k value for nearest neighbors
        
    """ Build Faiss index """
    def fit(self, X, y):
        d = X.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(X.astype(np.float32))
        self.y = y
        
    """ Find k-nearest neighbors """
    def neighbors(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        return indices
    
    """Predict label based on majority voting"""
    def predict(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices].astype(np.int32)
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
    
    """ Used for SV Object Recognition."""
    def predict_sv(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k+1)
        votes = self.y[indices].astype(np.int32)
        # View 0 is used as probe so it is excluded during predictions 
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes[:,1:]])
        return predictions

"""
===============================================================================
                             Inference framework
===============================================================================
"""


def predict_object_label(Query, Database, labels):
    CLS_KNN = FaissKNeighbors(1) # predict using 1-NN
    CLS_KNN.fit(np.array(Database), np.array(labels))
    obj_predict = CLS_KNN.predict(np.array(Query))
    return obj_predict

def predict_MVOR(refemb, testemb, testlabel, Config):
    """Compute multi-view object-level recognition accuracy"""
    acc = 0
    #print(len(refemb), len(testemb), len(testlabel))
    op = predict_object_label(testemb, refemb, testlabel)
    for x in tqdm(range(len(testlabel))):
        if op[x] == testlabel[x]:
            acc +=1
    print("MV Object Recognition Accuracy: ", acc/len(testlabel)*100)
    
    """Compute multi-view object-level retrieval mAP"""
    # Rank all objects in test database for a multi-view query of a particular object
    # Note: reference views and test views for each object are disjoint
    OR_KNN = FaissKNeighbors(Config.Ntest) 
    OR_KNN.fit(np.array(refemb), np.array(testlabel))
    objranks = OR_KNN.neighbors(np.array(testemb))
    obj_mAP = calculate_mAP_large(objranks, torch.tensor(testlabel))
    print("MV Object Retrieval mAP: ", obj_mAP*100)
    return acc/len(testlabel)*100, obj_mAP*100
    
def NNC_PIE_CLS(refC_emb, testO_emb, testC_emb, Config):
    cacc = 0
    oacc = 0
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest
    
    for i, x in enumerate(refC_emb):
        if x.ndim == 1:
            refC_emb[i]= x.reshape(1, x.shape[0])
    XTrain = np.concatenate(refC_emb,axis=0)
    oTrain = [[i]*refC_emb[i].shape[0] for i in range(Config.Ntrain)] 
    oTrain = torch.tensor(reduce(operator.concat, oTrain))
    cTrain = [[o2cTrain[i]]*refC_emb[i].shape[0] for i in range(Config.Ntrain)] 
    cTrain = torch.tensor(reduce(operator.concat, cTrain))
    
    XCTest = np.concatenate(testC_emb,axis=0)
    XOTest = np.concatenate(testO_emb,axis=0)
    oTest = [[i]*testO_emb[i].shape[0] for i in range(Config.Ntest)] 
    oTest = torch.tensor(reduce(operator.concat, oTest))
    cTest = [[o2cTest[i]]*testC_emb[i].shape[0] for i in range(Config.Ntest)] 
    cTest = torch.tensor(reduce(operator.concat, cTest))
    
    """Compute single-view object-level retrieval mAP
    For a given single-view query of an object retrieve other views of the same object"""
    OR_KNN = FaissKNeighbors(Config.Ntest*Config.N_G)
    OR_KNN.fit(np.array(XOTest), np.array(oTest))
    objranks = OR_KNN.neighbors(np.array(XOTest))
    obj_mAP = calculate_mAP_large(objranks, oTest, 'other_view')
    print("SV Object Retrieval mAP: ", obj_mAP*100)
    del OR_KNN
    del objranks
    
    """Compute single-view object recognition accuracy """
    OC_KNN = FaissKNeighbors(1)
    OC_KNN.fit(np.array(XOTest), np.array(oTest))
    objpredict = OC_KNN.predict_sv(np.array(XOTest))
    for x, gt in enumerate(oTest):
        if objpredict[x] == gt:
            oacc +=1
    print("SV Object Recognition Accuracy: ", oacc/len(oTest)*100)
    del OC_KNN
    del objpredict
    
    """Compute single-view category recognition accuracy """
    CLS_KNN = FaissKNeighbors(10) 
    CLS_KNN.fit(np.array(XTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))
    
    for t in range(0,len(cTest)):
        if cls_predict[t]==cTest[t]:
            cacc += 1
    print("SV Category Recognition Accuracy: ", cacc/len(cTest)*100)
        
    del CLS_KNN
    del cls_predict
    del XTrain
    del cTrain
    del oTrain
    
    """Compute single-view category retrieval mAP """
    CR_KNN = FaissKNeighbors(Config.Ntest*Config.N_G)
    CR_KNN.fit(np.array(XCTest), np.array(cTest))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP_large(crranks, cTest)
    print("SV Category Retrieval mAP: ", cr_mAP*100)
        
    del CR_KNN
    del crranks
    
    return obj_mAP*100, cr_mAP*100, cacc/len(cTest)*100, oacc/len(oTest)*100
    
def NNC_PIE_CLS_MV(refC_emb, testC_emb, Config):
    cacc = 0
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest
    XTrain = np.concatenate(refC_emb,axis=0)

    cTrain = torch.tensor([o2cTrain[i] for i in range(Config.Ntrain)])
    
    XCTest = np.concatenate(testC_emb,axis=0)
    cTest = torch.tensor([o2cTest[i] for i in range(Config.Ntest)])
    
    """Compute multi-view category retrieval mAP """
    CR_KNN = FaissKNeighbors(Config.Ntest)
    CR_KNN.fit(np.array(XCTest), np.array(cTest))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP_large(crranks, cTest)
    
    print("MV Category Retrieval mAP: ", cr_mAP*100)
        
    del CR_KNN
    del crranks
    
    """Compute multi-view category recognition accuracy """
    CLS_KNN = FaissKNeighbors(10) 
    CLS_KNN.fit(np.array(XTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))
    
    for t in range(0,len(cTest)):
        if cls_predict[t]==cTest[t]:
            cacc += 1
            
    print("MV Category Recognition Accuracy: ", cacc/len(cTest)*100)
        
    del CLS_KNN
    del cls_predict
    
    return cr_mAP*100, cacc/len(cTest)*100
"""
===============================================================================
Evaluate PI category and object-level classification and retrieval performance 
                            for Dual embedding space
===============================================================================
"""
def evaluate_performance_dual(dataset, Config, trcv_model, nview):
    ref_OE={}
    ref_CE={}
    mv_ref_CE={}
    test_OE={}
    test_CE={}
    label_obj = {}
    ref_mv_OE = {}
    test_mv_OE = {}
    label_mv_OE = {}
    """
    Load the Dual embedding space model and 
    extract the gallery (train) embeddings and probe (test) embeddings
    """
    trainData = loadDataset(Config, 'train', dataset)
    trainLoader = DataLoader(trainData,shuffle=False,num_workers=16,batch_size=1)
    trcv_model.eval()
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)): 
            rOE, rCE, mvrOE, mvrCE, _,_ = trcv_model(ref_data)
            ref_OE[i]=rOE.squeeze().detach().cpu().numpy()
            ref_CE[i]=rCE.squeeze().detach().cpu().numpy()
            mv_ref_CE[i]=mvrCE.detach().cpu().numpy()
    del trainData
    del trainLoader
    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData,shuffle=False,num_workers=16,batch_size=1)
    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)): 
            if nview == 1:
                """ if single-view query """
                tOE, tCE, _, _, _,_ = trcv_model(test_data)
                test_OE[j]=tOE.squeeze().detach().cpu().numpy()                
                test_CE[j]=tCE.squeeze().detach().cpu().numpy()
                label_obj[j]= obj_labels
                
            elif nview > 1:
                """ if multi-view query """
                tOE, tCE, mvtOE, mvtCE, _,_ = trcv_model(test_data)
                test_CE[j]=mvtCE.detach().cpu().numpy()
                """ 
                    evaluating if the model can extract multi-view embeddings where the 
                    reference images and the test query images of an unseen object are from disparate viewpoints                    
                    split all available views of test object into two disjoint sets: 
                    gallery: set comprising of reference views 
                    probe: set comprising of test views  
                """
                p = int(Config.N_G/2)
                _, _,refmvobj, _, _, _ = trcv_model(test_data[:,:p])
                _, _,testmvobj, _, _, _ = trcv_model(test_data[:,p:])
                ref_mv_OE[j]=refmvobj.squeeze().detach().cpu().numpy()
                test_mv_OE[j]=testmvobj.squeeze().detach().cpu().numpy()
                label_mv_OE[j]=j
    if nview == 1:
        """ Compute performance from single-view query """
        start2 = time.time()
        ormap, crmap, clsacc, svor  = NNC_PIE_CLS(list(ref_CE.values()), list(test_OE.values()), list(test_CE.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        return ormap, crmap, clsacc, svor
    if nview > 1:
        """ Compute performance from multi-view query """
        start2 = time.time()
        crmap, clsacc = NNC_PIE_CLS_MV(list(mv_ref_CE.values()), list(test_CE.values()), Config)
        mvor, mvoret = predict_MVOR(list(ref_mv_OE.values()), list(test_mv_OE.values()), list(label_mv_OE.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        return crmap, clsacc, mvor, mvoret

"""
===============================================================================
Evaluate PI category and object-level classification and retrieval performance 
                        for Single embedding space
===============================================================================
"""
def evaluate_performance_single(dataset, Config, trcv_model, nview):
    ref_OE={}
    ref_CE={}
    mv_ref_CE={}
    test_OE={}
    test_CE={}
    label_obj = {}
    ref_mv_OE = {}
    test_mv_OE = {}
    label_mv_OE = {}
    """
    Load the Single embedding space model and 
    extract the gallery (train) embeddings and probe (test) embeddings
    """
    trainData = loadDataset(Config, 'train', dataset)
    trainLoader = DataLoader(trainData,shuffle=False,num_workers=16,batch_size=1)
    trcv_model.eval()
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)): 
            rOE, mvrOE, _ = trcv_model(ref_data)
            ref_OE[i]=rOE.squeeze().detach().cpu().numpy()
            ref_CE[i]=rOE.squeeze().detach().cpu().numpy()
            mv_ref_CE[i]=mvrOE.detach().cpu().numpy()
    del trainData
    del trainLoader
    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData,shuffle=False,num_workers=16,batch_size=1)
    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)): 
            if nview == 1:
                """ if single-view query """
                tOE,_,_ = trcv_model(test_data)
                test_OE[j]=tOE.squeeze().detach().cpu().numpy()                
                test_CE[j]=tOE.squeeze().detach().cpu().numpy()
                label_obj[j]= obj_labels
                
            elif nview > 1:
                """ if multi-view query """
                tOE, mvtOE,_ = trcv_model(test_data)
                test_CE[j]=mvtOE.detach().cpu().numpy()
                """ 
                    evaluating if the model can extract multi-view embeddings where the 
                    reference images and the test query images of an unseen object are from disparate viewpoints                    
                    split all available views of test object into two disjoint sets: 
                    gallery: set comprising of reference views 
                    probe: set comprising of test views 
                """
                p = int(Config.N_G/2)
                _, refmvobj, _ = trcv_model(test_data[:,:p])
                _, testmvobj, _ = trcv_model(test_data[:,p:])
                ref_mv_OE[j]=refmvobj.squeeze().detach().cpu().numpy()
                test_mv_OE[j]=testmvobj.squeeze().detach().cpu().numpy()
                label_mv_OE[j]=j
    if nview == 1:
        """ Compute performance from single-view query """
        start2 = time.time()
        ormap, crmap, clsacc, svor  = NNC_PIE_CLS(list(ref_CE.values()), list(test_OE.values()), list(test_CE.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        return ormap, crmap, clsacc, svor
    if nview > 1:
        """ Compute performance from multi-view query """
        start2 = time.time()
        crmap, clsacc = NNC_PIE_CLS_MV(list(mv_ref_CE.values()), list(test_CE.values()), Config)
        mvor, mvoret = predict_MVOR(list(ref_mv_OE.values()), list(test_mv_OE.values()), list(label_mv_OE.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        return crmap, clsacc, mvor, mvoret