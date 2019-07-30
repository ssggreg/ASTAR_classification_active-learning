from Class_vogn import agent_cell
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import time
import torch.multiprocessing as mp
from active_function import *

class BaseNet(nn.Module):
    def __init__(self,dropout_rate=None):
        super(type(self), self).__init__()
        self.dropout = dropout_rate
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        self.layer1 = nn.Linear(8, 32)
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

    def __init__(self, in_size=8, hidden=64, dropout_rate=None):
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
    
Xtrain = np.load('eightPhasePowerStage_dc_dc_v2_lbl/trainx.npy')[:,1:7]
Ytrain = np.load('eightPhasePowerStage_dc_dc_v2_lbl/trainy.npy')*1.
Xtest = np.load('eightPhasePowerStage_dc_dc_v2_lbl/testx.npy')[:,1:7]
Ytest = np.load('eightPhasePowerStage_dc_dc_v2_lbl/testy.npy')*1.


for i in range(6):
    Xtrain[:,i]=(Xtrain[:,i] - Xtrain[:,i].mean())/(Xtrain[:,i].std())
    Xtest[:,i]=(Xtest[:,i] - Xtest[:,i].mean())/(Xtest[:,i].std())
Ytrain = Ytrain.reshape(-1,1)
Ytest = Ytest.reshape(-1,1) 

batch_size_sample = np.ones(261).astype(int)
for i in range(100):
    batch_size_sample[i+110]=3
for i in range(60):
    batch_size_sample[i+200]=5
    
batch_eval = np.ones(261).astype(int)*16

for i in range(100):
    batch_eval[i]=1
for i in range(60):
    batch_eval[i+100]=4
    
nb_ep = np.ones(261).astype(int)*50
for i in range(100):
    nb_ep[i+110]=30
for i in range(60):
    nb_ep[i+200]=20
    
nb_ech = 101
nb_batch=100

vogn_batch_size =batch_eval
loader_batch_size = vogn_batch_size 

functions = [f,max_entropy,bald,var_ratios,mean_std]

places_vogn = ['vogn_results/vogn_f.npy','vogn_results/vogn_ent.npy','vogn_results/vogn_bald.npy','vogn_results/vogn_var.npy','vogn_results/vogn_std.npy']

#attention VOG BIS

places_mc = ['mc_results/mc_f.npy','mc_results/mc_ent.npy','mc_results/mc_bald.npy','mc_results/mc_var.npy','mc_results/mc_std.npy']


if __name__ == '__main__':
    ttt = torch.zeros((5,1000))
    selection = torch.zeros((1+np.sum(batch_size_sample[:nb_batch]),1))
    print('vogn')
    mp.set_start_method('spawn')
    for j in range(0):
        processes = []
        print(functions[j])
        for i in range(5):
            torch.manual_seed(i)
            p = mp.Process(target=agent_cell,args=(Xtrain,Ytrain,Xtest,Ytest,BaseNet,EvalNet,functions[j],'VOGN',i,nb_batch,batch_size_sample,nb_ech,vogn_batch_size,batch_eval,nb_ep,ttt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
        np.save(places_vogn[j], ttt)
        
    ttt = torch.zeros((5,1000))
    print('mc')
    for j in range(1):
        processes = []
        print(functions[j])
        for i in range(1):
            torch.manual_seed(i)
            p = mp.Process(target=agent_cell,args=(Xtrain,Ytrain,Xtest,Ytest,BaseNet,EvalNet,functions[j],'MC',i,nb_batch,batch_size_sample,nb_ech,loader_batch_size,batch_eval,nb_ep,ttt,selection))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
        np.save(places_mc[j], ttt)
        print(selection)
        np.save('selection.npy', selection)