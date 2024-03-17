# -*- coding: utf-8 -*-
"""
Custom DataLoader for different datasets and other data related helper functions
"""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
from PIL import Image
import PIL.ImageOps  
import os
import torch
import numpy as np


"""
===============================================================================
    CUSTOM DATALOADERS FOR MULTI-VIEW TRAINING IMAGES FROM A PAIR OF OBJECTS
===============================================================================
self.transform = Config.train_dataAug # specifies transforms for data augmentation
self.gal_vp  = Config.gal_vp # specifies the gallery images 
self.N_vp = len(Config.gal_vp) # specifies the number of gallery images 
self.dpath = Config.gallery_dir # specifies the path for the gallery images
self.N_class = Config.Nclass # specifies the number of classes
self.N_comp = Config.Ncomp # For each object, this specifies the number of objects to compare with in each epoch

Methodology of sampling: 
__get_item__ function randomly samples one object (objx) and then randomly samples 
another object from the same category (objp). 
Then images of N_G views are randomly sampled from each of the objects and the class labels are returned
"""  

        
"""
Custom Dataset class for ObjectPI in the Wild (OOWL) Dataset
"""           
class OOWLTrainDataset(Dataset):
    
    def __init__(self,Config, simClass_list=[], name=""):
        self.transform = Config.train_dataAug 
        self.should_invert = False
        self.gal_vp  = Config.gal_vp
        self.N_vp = len(Config.gal_vp) 
        self.dpath = Config.gallery_dir
        self.N_class = Config.Ntrain
        self.N_comp = 10
        self.obj2cls = Config.o2ctrain
        self.cls2obj = Config.class_list
        #print(self.obj2cls)
        #print(self.cls2obj)
        self.dataset = name
        self.N_G = Config.N_G
        self.N_samples = 1
        print("Sample from same category !!!")
        
        
    def applyTransform(self,img_path):
        img = Image.open(img_path)
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img
        
    def __getitem__(self,index):
        # Get main class label from index
        objx = int(index/(self.N_comp*self.N_samples))
        cls = self.obj2cls[objx]
        allobjcls = self.cls2obj[cls].copy() ##### UNCOMMENT THIS
        #allobjcls = list(range(self.N_class)) #### COMMENT THIS (RANDOM SAMPLING)
        allobjcls.remove(objx)
        objp = random.choice(allobjcls)
        
        ximage = list()
        pimage = list()
        label = list()
        sampled_vp = random.sample(self.gal_vp, self.N_G)
        for i, vp in enumerate(sampled_vp):
            ximage.append(self.applyTransform(self.dpath+str(objx+1)+"/"+str(vp)+".jpg"))
            pimage.append(self.applyTransform(self.dpath+str(objp+1)+"/"+str(vp)+".jpg"))
            label.append(torch.from_numpy(np.array(cls)))
        return torch.stack(ximage), torch.stack(pimage), torch.stack(label)
        
    def __len__(self):
        return self.N_class*self.N_comp*self.N_samples
      
        
"""
Custom Dataset class for ModelNet40 Dataset
"""           
class MNet40TrainDataset(Dataset):
    
    def __init__(self,Config, simClass_list=[], name=""):
        self.transform = Config.train_dataAug 
        self.should_invert = False
        self.gal_vp  = Config.gal_vp
        self.N_vp = len(Config.gal_vp) 
        self.dpath = Config.gallery_dir
        self.N_class = Config.Ntrain
        self.N_comp = Config.Ncomp
        self.obj2cls = Config.o2ctrain
        self.cls2obj = Config.class_list
        self.dataset = name
        self.N_G = Config.N_G
        self.N_samples = 1
        print("Sample from same category !!!")
        
    def applyTransform(self,img_path):
        img = Image.open(img_path)
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img
        
    def __getitem__(self,index):
        objx = int(index/(self.N_comp*self.N_samples))
        cls = self.obj2cls[objx]
        allobjcls = self.cls2obj[cls].copy()
        #allobjcls = list(range(self.N_class)) #### COMMENT THIS (RANDOM SAMPLING)
        allobjcls.remove(objx)
        objp = random.choice(allobjcls)
        ximage = list()
        pimage = list()
        label = list()
        sampled_vp = random.sample(self.gal_vp, self.N_G)
        for i, vp in enumerate(sampled_vp):
            ximage.append(self.applyTransform(self.dpath+str(objx+1)+"/"+str(vp).zfill(3)+".jpg"))
            pimage.append(self.applyTransform(self.dpath+str(objp+1)+"/"+str(vp).zfill(3)+".jpg"))
            label.append(torch.from_numpy(np.array(cls)))
        return torch.stack(ximage), torch.stack(pimage), torch.stack(label)
        
    def __len__(self):
        return self.N_class*self.N_comp*self.N_samples

"""
Custom Dataset class for Fine-grained 3D (FG3D) Dataset
"""

class FG3DTrainDataset(Dataset):
    
    def __init__(self,Config, simClass_list=[], name=""):
        self.transform = Config.train_dataAug 
        self.should_invert = False
        self.gal_vp  = Config.gal_vp
        self.N_vp = len(Config.gal_vp) 
        self.dpath = Config.gallery_dir
        self.N_class = Config.Ntrain
        self.N_comp = Config.Ncomp
        self.obj2cls = Config.o2ctrain
        self.cls2obj = Config.class_list
        self.dataset = name
        self.N_G = Config.N_G
        self.N_samples = 1
        print("Sample from same category !!!")
        
    def applyTransform(self,img_path):
        img = Image.open(img_path)
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img
        
    def __getitem__(self,index):
        objx = int(index/(self.N_comp*self.N_samples))
        cls = self.obj2cls[objx]
        allobjcls = self.cls2obj[cls].copy()
        #allobjcls = list(range(self.N_class)) #### COMMENT THIS (RANDOM SAMPLING)
        allobjcls.remove(objx)
        objp = random.choice(allobjcls)
        ximage = list()
        pimage = list()
        label = list()
        sampled_vp = random.sample(self.gal_vp, self.N_G)
        for i, vp in enumerate(sampled_vp):
            ximage.append(self.applyTransform(self.dpath+str(objx+1)+"/"+str(vp).zfill(3)+".png"))
            pimage.append(self.applyTransform(self.dpath+str(objp+1)+"/"+str(vp).zfill(3)+".png"))
            label.append(torch.from_numpy(np.array(cls)))
        return torch.stack(ximage), torch.stack(pimage), torch.stack(label)
        
    def __len__(self):
        return self.N_class*self.N_comp*self.N_samples
    
"""
===============================================================================
          CUSTOM DATALOADER FOR TESTING, VALIDATION, AND EVALUATION
===============================================================================
Custom Dataset class for loading multi-view images of each object
"""  

class loadDataset(Dataset):
    
    def __init__(self,Config, split, dataset_name):
        self.transform = transforms.Compose([
                                       transforms.Resize((Config.imgDim, Config.imgDim)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ])
        self.split = split
        if self.split == "train" or self.split == "val":
            self.N_class = Config.Ntrain
            self.datadir = Config.gallery_dir
            self.obj2cls = Config.o2ctrain
        elif self.split == "test":
            self.N_class = Config.Ntest
            self.datadir = Config.probe_dir
            self.obj2cls = Config.o2ctest
        self.dataset = dataset_name
        self.N_G = Config.N_G
        self.Config = Config
        
    def applyTransform(self,img_path):
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img
            
    def __getitem__(self,index):
        obj_ind = index
        cls_ind = self.obj2cls[obj_ind]
        images = list()
        obj_labels = list()
        cls_labels = list()
        
        if self.split == 'train':
            vp = self.Config.gal_vp
        elif self.split == 'val':
            vp = self.Config.gal_vp
        elif self.split == 'test':
            vp = self.Config.probe_vp 
                
        for j in vp:
            if self.dataset == "OOWL":
                data_path = self.datadir+str(index+1)+"/"+str(j)+".jpg"
            elif self.dataset == "MNet40":
                data_path = self.datadir+str(index+1)+"/"+str(j).zfill(3)+".jpg"
            elif self.dataset == "FG3D":
                data_path = self.datadir+str(index+1)+"/"+str(j).zfill(3)+".png"
            else:
                print("Dataset not recognized.")
                
            images.append(self.applyTransform(data_path))
            obj_labels.append(torch.from_numpy(np.array(obj_ind)))
            cls_labels.append(torch.from_numpy(np.array(cls_ind)))
        return torch.stack(images), torch.stack(obj_labels), torch.stack(cls_labels)
        
    def __len__(self):
        return self.N_class
    
def load_class_data(i, dataset, datadir, flag, Config):
    temp = list()
    if flag == 0:
        vp = Config.gal_vp
    elif flag == 1:
        vp = Config.probe_vp 
    elif flag == 2:
        vp = Config.val_vp
    else:
            print("Error. Wrong Flag.")
    for j in vp:
        if dataset == "OOWL":
            data_path = datadir+str(i+1)+"/"+str(j)+".jpg"
        elif dataset == "MNet40":
            data_path = datadir+str(i+1)+"/"+str(j).zfill(3)+".jpg"
        elif dataset == "FG3D":
            data_path = datadir+str(i+1)+"/"+str(j).zfill(3)+".png"
        else:
            print("Dataset not recognized.")
        if os.path.isfile(data_path):
            img = Image.open(data_path)
            timg = transforms.Compose([
                                   transforms.Resize(Config.imgDim),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ])
            t  = timg(img)
            t.resize_((1,Config.inpChannel,Config.imgDim,Config.imgDim))
            temp.append(t)
    return temp

"""
Calculate statistics from the current embedding space such as:
inter-object and intra-object distances that is used for tracking performance 
of algorithm during training and used for checking early convergence criterion.
""" 
def calculate_stats(net, Config, dataset, emb_space = 'dual'):
    dismetric = []
    interCN_catg =[]
    interCN_min = []
    maxdisintra = np.zeros(Config.Ntrain)
    disintra = np.zeros(Config.Ntrain)
    ref_emb={}
    def distance_stats(i):
        ed_i = torch.cdist(ref_emb[i],ref_emb[i])
        maxdisintra[i] = torch.max(ed_i).data.cpu().numpy() # max-intra object distance
        disintra[i] = torch.mean(ed_i).data.cpu().numpy() # mean of pairwise distances between embeddings of same object
        category = Config.o2ctrain[i]
        objects = Config.class_list[category] # all objects from the same category as object i
        dismetric.append([])
        interCN_catg.append([])
        interCN_min.append([])
        interobj_cat = []
        """
        compute pairwise distances between embeddings of object i and j from 
        the same category
        """
        for j in objects:
            if j == i: 
                continue
            ed_ij = torch.cdist(ref_emb[i],ref_emb[j]) 
            interobj_cat.append(torch.mean(ed_ij).data.cpu().numpy())
        """
        store the average and the minimum distance between object i and other 
        objects from the same category
        """
        dismetric[i] = np.asarray(interobj_cat)
        interCN_catg[i] = np.mean(dismetric[i]) # average distance
        interCN_min[i] = np.min(dismetric[i]) # minimum distance

    """
    Using latest trained model until previous epoch to
    extract reference embeddings from gallery images
    """
    trainData = loadDataset(Config, 'val', dataset)
    trainLoader = DataLoader(trainData,shuffle=False,num_workers=16,batch_size=1)
    net.eval()
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)): 
            if emb_space == 'dual':
                rOE, _, _, _, _,_ = net(ref_data)
            elif emb_space == 'single':
                rOE, _, _ = net(ref_data)
            rOE = rOE.squeeze()
            if rOE.ndim == 1:
                ref_emb[i]= rOE.reshape(1, rOE.shape[0])
            else:
                ref_emb[i] =rOE

    for i in tqdm(range(Config.Ntrain)):
        distance_stats(i)
    #print(np.asarray(interCN_catg).shape, np.asarray(interCN_min).shape)
    distInfo = [np.mean(maxdisintra), np.mean(disintra), np.mean(interCN_catg), np.mean(interCN_min)]
    return distInfo
