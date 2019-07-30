from vogn_utils import csvDataset,ToTensor,test_perf_py
import torch
import torch.nn as nn
import torch.optim as optim
from vogn import VOGN
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

class EvalNet(nn.Module):
    def __init__(self, in_size=9, dropout_rate=None):
        super(type(self), self).__init__()
        self.dropout = dropout_rate
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        self.f1 = nn.Linear(in_size, 64)
        self.f2 = nn.Linear(64,  64)
        self.f3 = nn.Linear(64,  1)

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
    
    
Xtrain = np.load('opAmp_280nm_GF55_Mod_30P_lbl/trainx.npy')
Ytrain = np.load('opAmp_280nm_GF55_Mod_30P_lbl/trainy.npy')*1.
Xtest = np.load('opAmp_280nm_GF55_Mod_30P_lbl/testx.npy')
Ytest = np.load('opAmp_280nm_GF55_Mod_30P_lbl/testy.npy')*1.


for i in range(9):
    Xtrain[:,i]=(Xtrain[:,i] - Xtrain[:,i].mean())/(Xtrain[:,i].std())
    Xtest[:,i]=(Xtest[:,i] - Xtest[:,i].mean())/(Xtest[:,i].std())
Ytrain = Ytrain.reshape(-1,1)
Ytest = Ytest.reshape(-1,1)
    
    
rez_r={}
rezz_r={}
    
for i in range(50):
    seed(3+i)
    results =[np.random.choice(3420, 1+j,replace=False) for j in range(684)]
    rez_r["group" + str(i)] = results

for i in range(50):
    doop = []
    for j in rez_r["group" + str(i)]:

        doop.append(test_perf_py(Xtrain[j],Ytrain[j],Xtest,Ytest))
    rezz_r["group" + str(i)] = doop
    
tss=np.zeros((50,684))
for i in range(50):
    tss[i] = np.array(rezz_r["group" + str(i)])

    
np.save('random_even_faster', tss)

    


