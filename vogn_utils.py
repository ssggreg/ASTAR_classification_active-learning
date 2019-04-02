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

# Note : attention aux colums de file_A et file, columns en plus etc. . .
# Test wether GPUs are available
use_cuda = torch.cuda.is_available()
print("Using Cuda: %s" % use_cuda)

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


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(type(self), self).__init__()
        self.layer1 = nn.Linear(16, 64)
        self.layer2 = nn.Linear(64, 256)
        self.layer3 = nn.Linear(256, 64)
        self.layer4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        out = self.layer4(x)
        return out
    
                                                      
def train_model_bb(liste,model, dataloaders, criterion, optimizer, num_epochs=25):
    """
    Performs Training and Validation on test set on the given model using the specified optimizer
    :param model: (nn.Module) Model to be trained
    :param dataloaders: (list) train and test dataloaders
    :param criterion: Loss Function
    :param optimizer: Optimizer to be used for training
    :param num_epochs: Number of epochs to train the model
    :return: trained model, test and train metric history
    """
    trainloader, testloader = dataloaders
    for epoch in range(num_epochs):
        listee = liste.copy()
        d = listee.pop()
        model.train(True)
        print('Epoch[%d]:' % epoch)
        running_train_loss = 0.
        a=-1
        for i in (trainloader):
            a+=1
            if (a==d):
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
                if len(listee)!=0:
                    d = listee.pop()
                else :
                    break

            # Print Training Progress
            if a%500 == 100:
                train_accuracy = accuracy_bb(model, trainloader)
                print('Iteration[%d]:  Train Accuracy: %f ' % (a+1, train_accuracy))

        train_accuracy, train_loss = accuracy_bb(model, trainloader, criterion)
        print('## Epoch[%d], Train Loss: %f   &   Train Accuracy: %f' % (epoch, train_loss, train_accuracy))
    return model, train_loss, train_accuracy
      
def train_model_cc(model, dataloaders, criterion, optimizer, num_epochs=25):
    """
    Performs Training and Validation on test set on the given model using the specified
    :param model: (nn.Module) Model to be trained
    :param dataloaders: (list) train and test dataloaders
    :param criterion: Loss Function
    :param optimizer: Optimizer to be used for training
    :param num_epochs: Number of epochs to train the model
    :return: trained model, test and train metric history
    """
    trainloader, testloader = dataloaders
    for epoch in range(num_epochs):
        model.train(True)
        print('Epoch[%d]:' % epoch)
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
        print('## Epoch[%d], Train Loss: %f   &   Train Accuracy: %f' % (epoch, train_loss, train_accuracy))
    return model, train_loss, train_accuracy
      
      
def f (x):
    return x*(1-x)
      
      
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

#file_A=pd.read_csv('circuit-design/opAmp_280nm_GF55.csv')
#file = pd.read_csv('circuit-design/opAmp_280nm_GF55_Mod_30P.csv')
#b=['Pass/Fail']
#label = np.array(file[b].values=='Fail')*(-1.)+1
#b = []
#for i in list(file_A.columns):
    #if i != 'Pass/Fail':
        #b.append(i)
        #data = np.array(file[b].values)
#for i in range(16):
    #data[:,i]=(data[:,i] - data[:,i].mean())/(data[:,i].std())
    
#trainX_tmp, testX, trainY_tmp, testY = train_test_split(data, label, test_size=0.2,random_state=1)
#trainX_tmp, valX, trainY_tmp, valY = train_test_split(trainX_tmp, trainY_tmp, test_size=0.25, random_state=1)
#use_cuda = torch.cuda.is_available()                                                      
                                                      
#X = np.concatenate((trainX_tmp,valX,testX),axis=0)
#Y = np.concatenate((trainY_tmp,valY,testY),axis=0)
                                                      
                                                      

#file_dataset = csvDataset(X,Y,transform= ToTensor())
#dataset_loader = torch.utils.data.DataLoader(file_dataset,batch_size=1, shuffle=False)
#inference_loader = torch.utils.data.DataLoader(file_dataset,batch_size=6912, shuffle=False)

def reste (a):
    
    b = np.arange(4146)
    for i in a :
        b[i]=5000
    b = list(dict.fromkeys(b))
    b.pop()
    return b

def assist (b,a,count):
    c = 0
    d = b.shape
    d=4145
    while c != count:
        if b[d] not in a:
            a.append(b[d])
            c+=1
        d-=1
    return a
                                                      
def sampling_selection (depart,nb_ech,nb_ep,nb_batch,batch_size_sample,vogn_batch_size,class_model,X,Y):                                                
    a = depart
    start = time.time()
    inference_dataset = csvDataset(X,Y,transform= ToTensor())
    inference_loader = torch.utils.data.DataLoader(inference_dataset,batch_size=6920, shuffle=False)
                                                      
    for k in range(nb_batch):
        
        file_dataset = csvDataset(X[a],Y[a],transform= ToTensor())
        #b = reste(a)
        #inference_dataset = csvDataset(X[b],Y[b],transform= ToTensor())

        dataset_loader = torch.utils.data.DataLoader(file_dataset,batch_size=vogn_batch_size, shuffle=False)
        
        model = class_model
        if use_cuda:
            model = model.float().cuda()
        criterion = F.binary_cross_entropy_with_logits
        optimizer = VOGN(model, train_set_size=len(a), prec_init=100, num_samples=4)
                                                      
        model, train_loss, train_accuracy = train_model_cc(model, [dataset_loader, dataset_loader], criterion,
    optimizer, num_epochs=nb_ep)
                
        labz=torch.zeros(nb_ech,6912).cuda()
        predict = torch.zeros(nb_ech,6912).cuda()
        
        model.eval()
        with torch.no_grad():
            for i in range(nb_ech):
                predictions,lbl = inference_bb(model, inference_loader,optimizer,1)
                predict[i] = predictions.view(6912)
                labz[i] = lbl.view(6912)

        predict_train = np.sum(predict.cpu().numpy(),axis=0)/nb_ech    
        #BB =list(np.argsort(f(predict_train))[4146-batch_size:])+a
        a = assist(np.argsort(f(predict_train)),a,batch_size_sample)
        print(k)
                                                      
    end = time.time()
    print(end - start)
                                                      
    return a
              
              
              