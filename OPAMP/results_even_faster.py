from vogn_utils import sampling_selection_fast,csvDataset,ToTensor
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

rez_={}
rezz_={} 

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
    
for i in range(5):
    seed(i)
    depart = [np.random.choice(3420)]
    nb_ech = 101
    nb_batch=261
    class_model = SimpleConvNet()
    results = sampling_selection_fast (depart,nb_ech,nb_ep,nb_batch,batch_size_sample,vogn_batch_size,class_model,Xtrain,Ytrain,i)
    rez_["group" + str(i)] = results    
    
for i in range(5):
    doop = []
    for j in rez_["group" + str(i)]:
        model = SimpleConvNet()
        optimizer = optim.Adam(model.parameters(), weight_decay=0.001)
        doop.append(test_perf_py(model,optimizer,Xtrain[j],Ytrain[j],Xtest,Ytest))
    rezz_["group" + str(i)] = doop    
    
    
rez_r={}
rezz_r={}
    
for i in range(50):
    seed(3+i)
    results =[np.random.choice(3420, 1+j,replace=False) for j in range(684)]
    rez_r["group" + str(i)] = results

for i in range(50):
    doop = []
    for j in rez_r["group" + str(i)]:
        model = SimpleConvNet()
        optimizer = optim.Adam(model.parameters(), weight_decay=0.001)
        doop.append(test_perf_py(model,optimizer,Xtrain[j],Ytrain[j],Xtest,Ytest))
    rezz_r["group" + str(i)] = doop
    
tss=np.zeros((50,684))
for i in range(50):
    tss[i] = np.array(rezz_r["group" + str(i)])
    
ttt=np.zeros((5,262))
for i in range(5):
    ttt[i] =np.array(rezz_["group" + str(i)])
    
np.save('random_accuracies_1_100_step1_fast_even', tss)
np.save('VOGN_accuracies_1_81_step1_fast_even', ttt)

w = csv.writer(open("VOGN_selection_fast.csv_even", "w"))
for key, val in rez_.items():
    w.writerow([key, val])
    
w = csv.writer(open("Random_selection_fast_even.csv", "w"))
for key, val in rez_r.items():
    w.writerow([key, val])

nnn = np.ones(262)
for i in range(261):
    nnn[i+1] = nnn[i] + batch_size_sample[i]
    
plot_mean_std((np.arange(684)+1)/3420,tss,'random selection_fast_even')
plot_mean_std(nnn/3420,ttt,'VOGN selection_fast_even',color='y')
plt.savefig('graph_1_fast_even.png')    

