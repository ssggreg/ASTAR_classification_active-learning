import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp
from active_function import *


from Class_vogn_final import agent_cell

class BaseNet(nn.Module):
    def __init__(self,dropout_rate=None):
        super(type(self), self).__init__()
        self.dropout = dropout_rate
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        self.layer1 = nn.Linear(9, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 1)
        
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.layer2(x))
        if self.dropout:
            x = self.dropout(x)
        out = self.layer3(x)
        return out
    
class EvalNet(nn.Module):

    def __init__(self, in_size=9, hidden=64, dropout_rate=None):
        super(EvalNet, self).__init__()
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
        out = self.f3(out)
        return out    
    
Xtrain = np.load('opAmp_280nm_GF55_Mod_30P_lbl/trainx.npy')
Ytrain = np.load('opAmp_280nm_GF55_Mod_30P_lbl/trainy.npy')*1.
Xtest = np.load('opAmp_280nm_GF55_Mod_30P_lbl/testx.npy')
Ytest = np.load('opAmp_280nm_GF55_Mod_30P_lbl/testy.npy')*1.


for i in range(9):
    Xtrain[:,i]=(Xtrain[:,i] - Xtrain[:,i].mean())/(Xtrain[:,i].std())
    Xtest[:,i]=(Xtest[:,i] - Xtrain[:,i].mean())/(Xtrain[:,i].std())
Ytrain = Ytrain.reshape(-1,1)
Ytest = Ytest.reshape(-1,1) 

batch_size_sample = np.ones(261).astype(int)
for i in range(90):
    batch_size_sample[i+110]=3
for i in range(61):
    batch_size_sample[i+200]=5
    
vogn_batch_size = np.ones(261).astype(int)

loader_batch_size = vogn_batch_size 

for i in range(100):
    vogn_batch_size[i]=1
for i in range(60):
    vogn_batch_size[i+100]=4
    
nb_ep = np.ones(261).astype(int)*50
    
nb_ech = 10
nb_batch= 261



uncertainty = ['mc_dropout','bootstrap','vogn']

functions = [f,max_entropy,bald,var_ratios,mean_std]

functions_str = ['f','max_entropy','bald','var_ratios','mean_std']

functions_sec = ['no_secondary','ebmal']

recalibration = ['yes','no']


if __name__ == '__main__':
    
    mp.set_start_method('spawn')
    
    for j in range(1):
        
        j+=1

        for k in range(2):
                        
            for l in range(2):
                
                for r in range(2):

                
                    selection = torch.zeros((1+np.sum(batch_size_sample[:nb_batch]),1))
                    ttt = torch.zeros((5,1000))
                    processes = []        
                    
                    places = 'results_' + uncertainty[j] + '/' + functions_str[k] + '_' + functions_sec[l] + '_' + recalibration[r] + '.npy'

                    for i in range(5):

                        print(uncertainty[j])
                        print(functions[k])
                        print(functions_sec[l])
                        print(recalibration[r],' calibration')
                        print('seed',i)

                        torch.manual_seed(i)
                        
                        p = mp.Process(target=agent_cell,args=(Xtrain,Ytrain,Xtest,Ytest,BaseNet,EvalNet,uncertainty[j],functions[k],functions_sec[l],recalibration[r],i,nb_batch,batch_size_sample,nb_ech,loader_batch_size,nb_ep,ttt))
                        p.start()
                        processes.append(p)

                    for p in processes:
                        p.join()
                        
                    np.save(os.path.join(places), ttt)

        
        
        
        
        
        
        
        
        
        
        
        
