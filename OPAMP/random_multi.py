from vogn_utils import sampling_selection_fast_multi,csvDataset,ToTensor
from tf_utils import test_perf,test_perf_py
from mc_dropout import mc_dropout_selection
import torch
import torch.nn as nn
import torch.optim as optim
from vogn import VOGN
from datasets import Dataset
from utils import train_model
from utils import inference
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

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(type(self), self).__init__()
        
        self.layer1 = nn.Linear(9, 64)
        self.layer2 = nn.Linear(64, 256)
        self.layer3 = nn.Linear(256, 64)
        self.layer4 = nn.Linear(64, 1)
        
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        out = self.layer4(x)
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

rez_r={}
rezz_r={} 

def lll(i,Xtrain,Ytrain,Xtest,Ytest,rezz_r,rez_r):
    doop = []
    for j in rez_r["group" + str(i)]:
        model = SimpleConvNet()
        optimizer = optim.Adam(model.parameters(), weight_decay=0.001)
        doop.append(test_perf_py(model,optimizer,Xtrain[j],Ytrain[j],Xtest,Ytest))
            
    rezz_r["group" + str(i)] = doop       
            
if __name__ == '__main__': 
    
    
    for i in range(5):
        seed(3+i)
        results =[np.random.choice(3420, 1+j,replace=False) for j in range(684)]
        rez_r["group" + str(i)] = results
    
    
    mp.set_start_method('spawn')
    processes = []          

    for i in range(5):
        p = mp.Process(target=lll,args=(i,Xtrain,Ytrain,Xtest,Ytest,rezz_r,rez_r))       
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  

    tss=np.zeros((5,684))
    
    for i in range(5):
        tss[i] = np.array(rezz_r["group" + str(i)])


    np.save('random_even_faster_light', tss)


    w = csv.writer(open("Random_selection_fast_even_light.csv", "w"))
    for key, val in rez_r.items():
        w.writerow([key, val])

