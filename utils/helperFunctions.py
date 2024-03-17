# -*- coding: utf-8 -*-
"""
helper functions
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from csv import writer
import random 


def expanded_pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if y is not None:
         differences = x.unsqueeze(1) - y.unsqueeze(0)
    else:
        differences = x.unsqueeze(1) - x.unsqueeze(0)
    distances = torch.sum(differences * differences, -1)
    return distances
    
def blocks(vs,n):
    vs_blocks =[]
    length = int(np.ceil(len(vs)/n))
    for i in range(0,len(vs),length):
        vs_blocks.append(vs[i:i+length])
    return vs_blocks

def partition (list_in, n):
    random.shuffle(list_in)
    val_list = sorted(list_in[0:n])
    test_list = sorted(list_in[n:])
    return val_list, test_list

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def plot_ConvergenceCurve(train_loss_history, test_acc_history, path):
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(train_loss_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Test Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(test_acc_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(path)

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def plot_infoex(infoEx, path):
    fig, ax1 = plt.subplots()
    if len(infoEx) > 1:
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Informative Example Percentage', color=color)
        ax1.plot(infoEx[1:], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        fig.savefig(path)
        
def plot_distance(infoEx, max_intra, min_inter, ratio, path):
    fig, ax1 = plt.subplots()
    if len(infoEx) > 1:
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Ratio', color=color)
        ax1.plot(ratio[1:], color=color, label='Ratio')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Average Distance', color=color)  # we already handled the x-label with ax1
        
        #ax2.plot(inter[1:], 'b:', label='Inter-class (10 NN)')
        ax2.plot(max_intra[1:], 'g', label='Max Intra-class')
        #ax2.plot(intra[1:], 'g:', label='Avg Intra-class')
        ax2.plot(min_inter[1:], 'b', label='Min Inter-class')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #plt.show()
        
        legend1 = ax1.legend(loc=0)
        
        legend2 = ax2.legend()
        # Put a nicer background color on the legend.
        legend2.get_frame().set_facecolor('C0')
        
        plt.savefig(path)
