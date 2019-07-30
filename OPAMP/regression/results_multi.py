from vogn_utils import final_fast_multi,csvDataset,ToTensor
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

#class SimpleConvNet(nn.Module):
  #  def __init__(self):
   #     super(type(self), self).__init__()
    #    self.layer1 = nn.Linear(9, 64)
     #   self.layer2 = nn.Linear(64, 256)
      #  self.layer3 = nn.Linear(256, 64)
       # self.layer4 = nn.Linear(64, 1)

   # def forward(self, x):
    #    x = F.relu(self.layer1(x))
     #   x = F.relu(self.layer2(x))
      #  x = F.relu(self.layer3(x))
       # out = self.layer4(x)
        #return out
    
class SimpleConvNet(nn.Module):
    def __init__(self, in_size=9, hidden=64, dropout_rate=None):
        super(type(self), self).__init__()
        self.dropout = dropout_rate
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        self.f1 = nn.Linear(in_size, hidden)
        self.f2 = nn.Linear(hidden,  hidden)
        self.f3 = nn.Linear(hidden,  1)

    def forward(self, x):
        out = x
        out = F.relu(self.f1(out))
        if self.dropout:
            out = self.dropout(out)
        out = F.relu(self.f2(out))
        if self.dropout:
            out = self.dropout(out)
        out =self.f3(out)
        return out       
    
Xtrain = np.load('trainx.npy')
Ytrain = np.load('trainy.npy')*1.
Xtest = np.load('testx.npy')
Ytest = np.load('testy.npy')*1.


for i in range(9):
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

if __name__ == '__main__':
    ttt = torch.zeros((5,101))
    mp.set_start_method('spawn')
    processes = []
    for i in range(5):
        seed(i)
        depart = [np.random.choice(3420)]
        nb_ech = 101
        nb_batch=100
        class_model = SimpleConvNet()
        p = mp.Process(target=final_fast_multi,args=(depart,nb_ech,nb_ep,nb_batch,batch_size_sample,vogn_batch_size,class_model,Xtrain,Ytrain,Xtest,Ytest,i,ttt,f))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    print (ttt)

    np.save('aaab', ttt)


