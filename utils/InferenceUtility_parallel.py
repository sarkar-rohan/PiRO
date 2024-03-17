#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for inference on Pose-invariant Classification and Retrieval (PICR) 
category and object-level tasks. This is paralleized to use all CPU cores but
still slow for large datasets such as FG3D. 
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
from utils.rank_metrics import calculate_mAP, average_precision
from utils.DataUtility_PiRO import loadDataset
import time
from joblib import Parallel, delayed
import os
print("Inference Utility - Parallelized")
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
        votes = self.y[indices]
        return votes
    
    """Predict label based on majority voting"""
    def predict(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices].astype(np.int32)
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

"""
===============================================================================
                             Inference framework
===============================================================================
"""


def predict_object_label(Query, Database, labels):
    """ Predict object label """
    CLS_KNN = FaissKNeighbors(1) # predict using 1-NN
    CLS_KNN.fit(np.array(Database), np.array(labels))
    obj_predict = CLS_KNN.predict(np.array(Query))
    return obj_predict

def retrieve_object_otherviews(Query, Database, labels):
    """ Retrieve other views of the same object-identity as query """
    RET_KNN = FaissKNeighbors(len(Database))
    RET_KNN.fit(np.array(Database), np.array(labels))
    objranks = RET_KNN.neighbors(np.array(Query))
    return objranks

def predict_MV_object_tasks(refemb, testemb, testlabel, Config):
    """Compute multi-view object-level recognition accuracy"""
    acc = 0
    op = predict_object_label(testemb, refemb, testlabel)
    for x in tqdm(range(len(testlabel))):
        if op[x] == testlabel[x]:
            acc +=1
    print("MV Object Recognition Accuracy: ", acc/len(testlabel))
    
    """Compute multi-view object-level retrieval mAP"""
    # Rank all objects in test database for a multi-view query of a particular object
    # Note: reference views and test views for each object are disjoint
    OR_KNN = FaissKNeighbors(Config.Ntest) 
    OR_KNN.fit(np.array(refemb), np.array(testlabel))
    objranks = OR_KNN.neighbors(np.array(testemb))
    obj_mAP = calculate_mAP(objranks, torch.tensor(testlabel))
    print("MV Object Retrieval mAP: ", obj_mAP*100)
    return acc/len(testlabel)*100, obj_mAP*100
    

"""
def predict_SV_object_tasks(testemb, testlabel, Config):
    # Not parallelized, very slow for large datasets such as FG3D
    acc = 0
    ret_otherviews = []
    XOTest = np.concatenate(testemb,axis=0)
    oTest = np.concatenate(np.squeeze(np.asarray(testlabel)),axis=0)
    for k in tqdm(range(len(oTest))):
        database = list(XOTest.copy())
        labels = list(oTest.copy())
        probe = [XOTest[k]]
        gt = oTest[k]
        del database[k]
        del labels[k]
        #Compute single-view object recognition accuracy
        op_cls = predict_object_label(probe, database, labels)
        if op_cls == gt:
            acc +=1
        # Compute single-view object-level retrieval mAP
        # For a given single-view query of an object retrieve other views of the same object
        op_ret = retrieve_object_otherviews(probe, database, labels)
        ret_otherviews.append(op_ret)
    obj_mAP = calculate_mAP(np.squeeze(np.asarray(ret_otherviews)), torch.tensor(oTest))
    print("SV Object Recognition Accuracy: ", acc/len(oTest)*100)
    print("SV Object Retrieval mAP: ", obj_mAP*100)
    return acc/len(oTest)*100, obj_mAP*100
"""

def predict_SV_split(XOTest, oTest, k):
    """ Compute performance on object-level tasks for a split where one view 
        of an object is selected as the probe and the remaining test dataset is 
        selected as the gallery """
    gallery = list(XOTest.copy())
    labels = list(oTest.copy())
    probe = [XOTest[k]] # select k th single-view image of an object as the probe
    gt = oTest[k] # ground-truth label of the probe
    del gallery[k] # the rest of the test dataset is considered as the gallery 
    del labels[k]
    """Compute single-view object recognition accuracy for a probe image"""
    op_cls = predict_object_label(probe, gallery, labels)
    if op_cls == gt:
        acc = 1
    else:
        acc = 0
    """Compute single-view object-level retrieval mAP
    For a given single-view query of an object retrieve other views of the same object"""
    op_ret = retrieve_object_otherviews(probe, gallery, labels)
    rank = op_ret[0,:] == gt 
    ap = average_precision(rank)
    return [acc, ap]
def predict_SV_object_tasks(testemb, testlabel, Config):
    """ Compute performance on object-level tasks for all splits where one view 
        of an object is selected as the probe and the remaining test dataset is 
        selected as the gallery. Computation for each split is paralleized 
        across all available CPU cores. This is still slow for large datasets 
        such as FG3D. """
    XOTest = np.concatenate(testemb,axis=0)
    oTest = np.concatenate(np.squeeze(np.asarray(testlabel)),axis=0)
    # repeat by selecting each view of each object as probe and the remaining test dataset as gallery 
    results = Parallel(n_jobs=os.cpu_count())(delayed(predict_SV_split)(XOTest, oTest, k) for k in tqdm(range(len(oTest))))
    sv_results = np.asarray(results)
    """ Compute average recognition accuracy and mAP for all splits """
    acc_list = sv_results[:,0]
    ap_list = sv_results[:,1]
    print("SV Object Recognition Accuracy: ", np.mean(acc_list)*100)
    print("SV Object Retrieval mAP: ", np.mean(ap_list)*100)
    return np.mean(acc_list)*100, np.mean(ap_list)*100

    
def predict_SV_category_tasks(refC_emb, testC_emb, Config):
    cacc = 0
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest
    
    for i, x in enumerate(refC_emb):
        if x.ndim == 1:
            refC_emb[i]= x.reshape(1, x.shape[0])
    XTrain = np.concatenate(refC_emb,axis=0)
    cTrain = [[o2cTrain[i]]*refC_emb[i].shape[0] for i in range(Config.Ntrain)] 
    cTrain = torch.tensor(reduce(operator.concat, cTrain))
    
    XCTest = np.concatenate(testC_emb,axis=0)
    cTest = [[o2cTest[i]]*testC_emb[i].shape[0] for i in range(Config.Ntest)] 
    cTest = torch.tensor(reduce(operator.concat, cTest))
    
    """Compute single-view category recognition accuracy """
    CLS_KNN = FaissKNeighbors(10) 
    CLS_KNN.fit(np.array(XTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))
    
    for t in range(0,len(cTest)):
        if cls_predict[t]==cTest[t]:
            cacc += 1
    print("SV Category Recognition: ", cacc/len(cTest))
        
    del CLS_KNN
    del cls_predict
    del XTrain
    del cTrain
    
    """Compute single-view category retrieval mAP """
    CR_KNN = FaissKNeighbors(Config.Ntest*Config.N_G)
    CR_KNN.fit(np.array(XCTest), np.array(cTest))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP(crranks, cTest)
    print("SV Category Retrieval mAP: ", cr_mAP)
        
    del CR_KNN
    del crranks
    
    return cr_mAP*100, cacc/len(cTest)*100
    
def predict_MV_category_tasks(refC_emb, testC_emb, Config):
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
    cr_mAP = calculate_mAP(crranks, cTest)
    
    print("MV Category Retrieval mAP: ", cr_mAP)
        
    del CR_KNN
    del crranks
    
    """Compute multi-view category recognition accuracy """
    CLS_KNN = FaissKNeighbors(10) 
    CLS_KNN.fit(np.array(XTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))
    
    for t in range(0,len(cTest)):
        if cls_predict[t]==cTest[t]:
            cacc += 1
            
    print("MV Category Recognition mAP: ", cacc/len(cTest))
        
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
                label_obj[j]= obj_labels.cpu().numpy()
                
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
        crmap, clsacc  = predict_SV_category_tasks(list(ref_CE.values()), list(test_CE.values()), Config)
        svor, ormap = predict_SV_object_tasks(list(test_OE.values()), list(label_obj.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        return ormap, crmap, clsacc, svor
    if nview > 1:
        """ Compute performance from multi-view query """
        start2 = time.time()
        crmap, clsacc = predict_MV_category_tasks(list(mv_ref_CE.values()), list(test_CE.values()), Config)
        mvor, mvoret = predict_MV_object_tasks(list(ref_mv_OE.values()), list(test_mv_OE.values()), list(label_mv_OE.values()), Config)
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
                label_obj[j]= obj_labels.cpu().numpy()
                
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
        crmap, clsacc  = predict_SV_category_tasks(list(ref_CE.values()), list(test_CE.values()), Config)
        svor, ormap = predict_SV_object_tasks(list(test_OE.values()), list(label_obj.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        return ormap, crmap, clsacc, svor
    if nview > 1:
        """ Compute performance from multi-view query """
        start2 = time.time()
        crmap, clsacc = predict_MV_category_tasks(list(mv_ref_CE.values()), list(test_CE.values()), Config)
        mvor, mvoret = predict_MV_object_tasks(list(ref_mv_OE.values()), list(test_mv_OE.values()), list(label_mv_OE.values()), Config)
        end2 = time.time()
        print("Time(Inference):", (end2-start2))
        return crmap, clsacc, mvor, mvoret