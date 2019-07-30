import torch
import torch.nn as nn
import torch.optim as optim
from vogn import VOGN
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn.functional as F
sns.set()
from torch.utils.data.dataloader import DataLoader
import time
from sklearn.model_selection import train_test_split

use_cuda = torch.cuda.is_available()


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


def test_perf_py(Xtr,Ytr,Xte,Yte):
    
    model = EvalNet()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    
    use_cuda = torch.cuda.is_available()
    inference_dataset = csvDataset(Xte,Yte,transform= ToTensor())
    inference_loader = torch.utils.data.DataLoader(inference_dataset,batch_size=Xte.shape[0], shuffle=False)
    file_dataset = csvDataset(Xtr,Ytr,transform= ToTensor())
    final_loader = torch.utils.data.DataLoader(file_dataset,batch_size=Xtr.shape[0], shuffle=False)
    
    if use_cuda:
        model = model.float().cuda()
        
    criterion = F.binary_cross_entropy_with_logits
    model, train_loss, train_accuracy = train_model_cc_fast(model, [final_loader, final_loader], criterion,
    optimizer, num_epochs=50)
    model.eval()
    with torch.no_grad():
        for i in inference_loader:
            inputs = i['data']
            labels = i['label']
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            out = model.forward(inputs)
            pred = (out.cpu().numpy()>0)*1.
            labels = (labels.cpu().numpy())*1.
        
    correct =(np.sum(pred==labels)/Xte.shape[0])
    
    return correct


class csvDataset(Dataset):
    def __init__(self, data,label, transform=None):
        self.label = label
        self.data = data
        #self.train_set = TensorDataset()
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        sample = { 'data': data,'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
class ToTensor(object):
    def __call__(self, sample):
        data, label= sample['data'],sample['label']
        return {'data': torch.from_numpy(data).float(),'label': torch.from_numpy(label).float()}

    
def accuracy_bb(model, dataloader, criterion=None):
    """ Computes the model's classification accuracy on the train dataset
    Computes classification accuracy and loss(optional) on the test dataset
    The model should return logits
    """
    model.eval()
    with torch.no_grad():
        correct = 0.
        running_loss = 0.
        for i in (dataloader):
            inputs = i['data']
            labels = i['label']
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            pred = (outputs>0).float()
            correct += (pred.view(-1,1) == labels).sum().item()
        accuracy = correct / len(dataloader.dataset)
        if criterion is not None:
            running_loss = running_loss / len(dataloader)
            return accuracy, loss
    return accuracy


                                                      
def inference_bb(model, data_loader, optimizer,mc_samples):
    a=0
    for i in (data_loader):
            inputs = i['data']
            labels = i['label']
            
    if use_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()
                
    return optimizer.get_mc_predictions(model.forward,inputs,mc_samples=mc_samples)[0]>0,labels

def inference_pp(model, data_loader, optimizer,mc_samples):
    a=0
    for i in (data_loader):
            inputs = i['data']
            labels = i['label']
            
    if use_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()
                
    return optimizer.get_mc_predictions(model.forward,inputs,mc_samples=mc_samples)[0],labels

def inference_(model, data_loader, optimizer,mc_samples):

    for i in (data_loader):
            inputs = i['data']
            labels = i['label']
            inputs, labels = inputs.cuda(), labels.cuda()
                
    return model.forward(inputs),labels


def reste (a):
    
    b = np.arange(3420)
    for i in a :
        b[i]=5000
    b = list(dict.fromkeys(b))
    b.pop()
    return b

def assist (b,a,count):
    c = 0
    d = b.shape
    d=3419
    while c != count:
        if b[d] not in a:
            a.append(b[d])
            c+=1
        d-=1
    return a
                                                      

              
def train_model_cc_fast(model, dataloaders, criterion, optimizer, num_epochs=25):

    trainloader, testloader = dataloaders
    for epoch in range(num_epochs):
        model.train(True)
        #print('Epoch[%d]:' % epoch)
        running_train_loss = 0.
        a=-1
        for i in (trainloader):
            a+=1
            inputs = i['data']
            labels = i['label']
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            if isinstance(optimizer, VOGN):
                def closure():
                    optimizer.zero_grad()
                    logits = model.forward(inputs)
                    loss = criterion(logits, labels)
                    return loss
            else:
                def closure():
                    optimizer.zero_grad()
                    logits = model.forward(inputs)
                    loss = criterion(logits, labels)
                    loss.backward()
                    return loss
            loss = optimizer.step(closure)
            running_train_loss += loss.detach().item()
            # Print Training Progress
            #if a%500 == 100:
                #train_accuracy = accuracy_bb(model, trainloader)
                #print('Iteration[%d]:  Train Accuracy: %f ' % (a+1, train_accuracy))

        train_accuracy, train_loss = accuracy_bb(model, trainloader, criterion)
        #print('## Epoch[%d], Train Loss: %f   &   Train Accuracy: %f' % (epoch, train_loss, train_accuracy))
        
        if (epoch > 1) & (train_accuracy >0.999):
                break
    return model, train_loss, train_accuracy 


def final_fast_multi_vogn(depart,nb_ech,nb_ep,nb_batch,batch_size_sample,vogn_batch_size,class_model,X,Y,Xtest,Ytest,seeds,ttt,f):
    
    a = depart
    inference_dataset = csvDataset(X,Y,transform= ToTensor())
    inference_loader = torch.utils.data.DataLoader(inference_dataset,batch_size=3420, shuffle=False)

    results=[]
    results.append(a[:])
    start = time.time()


    for k in range(nb_batch):

        file_dataset = csvDataset(X[a],Y[a],transform= ToTensor())
        #b = reste(a)
        #inference_dataset = csvDataset(X[b],Y[b],transform= ToTensor())

        dataset_loader = torch.utils.data.DataLoader(file_dataset,np.asscalar(vogn_batch_size[k]), shuffle=False)

        model = class_model
        if use_cuda:
            model = model.float().cuda()
        criterion = F.binary_cross_entropy_with_logits
        optimizer = VOGN(model, train_set_size=len(a), prec_init=100, num_samples=4)

        model, train_loss, train_accuracy = train_model_cc_fast(model, [dataset_loader, dataset_loader], criterion,
    optimizer, num_epochs=nb_ep[k])

        labz=torch.zeros(nb_ech,3420).cuda()
        predict = torch.zeros(nb_ech,3420).cuda()
        model.eval()
        with torch.no_grad():
            for i in range(nb_ech):
                predictions,lbl = inference_pp(model, inference_loader,optimizer,1)
                predict[i] = predictions.view(3420)
                labz[i] = lbl.view(3420)

        predict_train = predict.cpu().numpy()
        #BB =list(np.argsort(f(predict_train))[4146-batch_size:])+a
        a = assist(np.argsort(f(predict_train)),a,batch_size_sample[k])
        results.append(a[:])
        print("batch",k,"seed",seeds)
                                                      
    
    i = 0
    for j in results:
        ttt[seeds,i] = test_perf_py(X[j],Y[j],Xtest,Ytest)
        i+=1
        
        
        
def inference_(model, data_loader, optimizer,mc_samples):

    for i in (data_loader):
            inputs = i['data']
            labels = i['label']
            inputs, labels = inputs.cuda(), labels.cuda()
                
    return model.forward(inputs),labels



def final_fast_multi_mc(depart,nb_ech,nb_ep,nb_batch,batch_size_sample,loader_batch_size,class_model,X,Y,Xtest,Ytest,seeds,ttt,f):
    a = depart
    inference_dataset = csvDataset(X,Y,transform= ToTensor())
    inference_loader = torch.utils.data.DataLoader(inference_dataset,batch_size=3420, shuffle=False)

    results=[]
    results.append(a[:])
    start = time.time()


    for k in range(nb_batch):

        file_dataset = csvDataset(X[a],Y[a],transform= ToTensor())
        #b = reste(a)
        #inference_dataset = csvDataset(X[b],Y[b],transform= ToTensor())

        dataset_loader = torch.utils.data.DataLoader(file_dataset,np.asscalar(loader_batch_size[k]), shuffle=False)

        model = class_model
        if use_cuda:
            model = model.float().cuda()
        criterion = F.binary_cross_entropy_with_logits
        optimizer = optim.Adam(model.parameters())

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

        predict_train = predict.cpu().numpy()
        #BB =list(np.argsort(f(predict_train))[4146-batch_size:])+a
        a = assist(np.argsort(f(predict_train)),a,batch_size_sample[k])
        results.append(a[:])
        print("batch",k,"seed",seeds)
                                                      
    
    i = 0
    for j in results:
        ttt[seeds,i] = test_perf_py(X[j],Y[j],Xtest,Ytest)
        i+=1

        