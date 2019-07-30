import torch
import torch.nn as nn
import torch.optim as optim
from vogn import VOGN
from models import SimpleConvNet
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
from vogn_utils import train_model_cc_fast,train_model_bb,csvDataset,assist,ToTensor

def f (x):
    return x*(1-x)

def inference_(model, data_loader, optimizer,mc_samples):
    use_cuda = torch.cuda.is_available()

    for i in (data_loader):
            inputs = i['data']
            labels = i['label']
            
    if use_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()
                
    return model.forward(inputs)>0,labels



def mc_dropout_selection(depart,nb_ech,nb_ep,nb_batch,batch_size_sample,loader_batch_size,class_model,X,Y,seeds):
    
    a = depart
    results=[a[:]]
    use_cuda = True
    start = time.time()
    inference_dataset = csvDataset(X,Y,transform= ToTensor())
    inference_loader = torch.utils.data.DataLoader(inference_dataset,batch_size=3420, shuffle=False)
                                                                                                     
    start = time.time()
                                                      
    for k in range(nb_batch):
        
        file_dataset = csvDataset(X[a],Y[a],transform= ToTensor())
        dataset_loader = torch.utils.data.DataLoader(file_dataset,batch_size=np.asscalar(loader_batch_size[k]), shuffle=False)
        model = class_model
        if use_cuda:
            model = model.float().cuda()
        criterion = F.binary_cross_entropy_with_logits
        optimizer = optim.Adam(model.parameters())
        model.dropout.requires_grad = False
                                                      
        model, train_loss, train_accuracy = train_model_cc_fast(model, [dataset_loader, dataset_loader], criterion,
    optimizer, num_epochs=nb_ep[k])
        
        labz=torch.zeros(nb_ech,3420).cuda()
        predict = torch.zeros(nb_ech,3420).cuda()
        
        model.train()
        with torch.no_grad():
            for i in range(nb_ech):
                predictions,lbl = inference_(model, inference_loader,optimizer,1)
                predict[i] = predictions.view(3420)
                labz[i] = lbl.view(3420)

        predict_train = np.sum(predict.cpu().numpy(),axis=0)/nb_ech    
        #BB =list(np.argsort(f(predict_train))[4146-batch_size:])+a
        a = assist(np.argsort(f(predict_train)),a,batch_size_sample[k])
        results.append(a[:])                                           

        print("batch",k,"seed",seeds)
                                                      
    end = time.time()
    print(end - start)
                                                      
    return results