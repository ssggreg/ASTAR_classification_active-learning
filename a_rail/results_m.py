from vogn_utils import final_fast_multi_vogn,csvDataset,ToTensor,final_fast_multi_mc
#from mc_dropout import mc_dropout_selection
import torch
import torch.nn as nn
import torch.optim as optim
from vogn import VOGN
#from datasets import Dataset
#from utils import train_model
#from utils import inference
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn.functional as F
sns.set()
from torch.utils.data.dataloader import DataLoader
import time
from sklearn.model_selection import train_test_split
from numpy.random import seed
import csv
import torch.multiprocessing as mp
from active_function import *

class SimpleConvNet(nn.Module):
    def __init__(self, in_size=16, dropout_rate=None):
        super(type(self), self).__init__()
        self.dropout = dropout_rate
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        self.f1 = nn.Linear(in_size, 64)
        self.f2 = nn.Linear(64,  128)
        self.f3 = nn.Linear(128,  128)
        self.f4 = nn.Linear(128,  1)

    def forward(self, x):
        out = x
        out = F.relu(self.f1(out))
        if self.dropout:
            out = self.dropout(out)
        out = F.relu(self.f2(out))
        if self.dropout:
            out = self.dropout(out)
        out =self.f3(out)
        out =self.f4(out)
        return out  
    
    
Xtrain = np.load('Rail2Rail_OPA_v1/trainx.npy')
Ytrain = np.load('Rail2Rail_OPA_v1/trainy.npy')*1.
Xtest = np.load('Rail2Rail_OPA_v1/testx.npy')
Ytest = np.load('Rail2Rail_OPA_v1/testy.npy')*1.


for i in range(16):
    Xtrain[:,i]=(Xtrain[:,i] - Xtrain[:,i].mean())/(Xtrain[:,i].std())
    Xtest[:,i]=(Xtest[:,i] - Xtest[:,i].mean())/(Xtest[:,i].std())
Ytrain = Ytrain.reshape(-1,1)
Ytest = Ytest.reshape(-1,1) 

#for i in range(5):
    #rez_["group" + str(i)]=torch.zeros((5,1+1)).cuda()
    #rez_["group" + str(i)]=torch.zeros((nb_seeds,1+nb_batch)).cuda()
batch_size_sample = np.ones(261).astype(int)
for i in range(100):
    batch_size_sample[i+110]=3
for i in range(60):
    batch_size_sample[i+200]=5
    
vogn_batch_size = np.ones(261).astype(int)
for i in range(100):
    vogn_batch_size[i+110]=16
for i in range(60):
    vogn_batch_size[i+200]=32
    
nb_ep = np.ones(261).astype(int)*50
for i in range(100):
    nb_ep[i+110]=10
for i in range(60):
    nb_ep[i+200]=5
loader_batch_size =  vogn_batch_size 
#mp.set_start_method('spawn')
 #   processes = []
  #  
   # for ACQUISITION_FCN in ACQUISITION_FCNS:
    #    for SEED in SEEDS:
     #       p = mp.Process(target=target, args=args)
      #      p.start()
       #     processes.append(p)
#
 #   for p in processes:
  #      p.join()
#ttt= np.zeros((5,262))

functions = [f,max_entropy,bald,var_ratios,mean_std]

places_vogn = ['vogn_results/vogn_f.npy','vogn_results/vogn_ent.npy','vogn_results/vogn_bald.npy','vogn_results/vogn_var.npy','vogn_results/vogn_std.npy']

places_mc = ['mc_results/mc_f.npy','mc_results/mc_ent.npy','mc_results/mc_bald.npy','mc_results/mc_var.npy','mc_results/mc_std.npy']


if __name__ == '__main__':
    ttt = torch.zeros((5,262))
    print('vogn')
    mp.set_start_method('spawn')
    for j in range(5):
        #break:
        processes = []
        print(functions[j])
        for i in range(5):
            seed(i)
            depart = [np.random.choice(9472)]
            nb_ech = 101
            nb_batch=261
            class_model = SimpleConvNet()
            p = mp.Process(target=final_fast_multi_vogn,args=(depart,nb_ech,nb_ep,nb_batch,batch_size_sample,vogn_batch_size,class_model,Xtrain,Ytrain,Xtest,Ytest,i,ttt,functions[j]))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        np.save(places_vogn[j], ttt)
    print('mc')    
    for j in range(5):
        processes = []
        print(functions[j])
        for i in range(5):
            seed(i)
            depart = [np.random.choice(9472)]
            nb_ech = 101
            nb_batch=261
            class_model = SimpleConvNet()
            p = mp.Process(target=final_fast_multi_mc,args=(depart,nb_ech,nb_ep,nb_batch,batch_size_sample,loader_batch_size,class_model,Xtrain,Ytrain,Xtest,Ytest,i,ttt,functions[j]))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        np.save(places_mc[j], ttt)


