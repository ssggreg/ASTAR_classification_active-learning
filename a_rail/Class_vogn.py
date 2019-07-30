import torch
import torch.nn as nn
import torch.optim as optim
from vogn import VOGN
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader


def agent_cell(Xpool,Ypool,Xtest,Ytest,Basenet,Evalnet,acquisition_f,optimizer,seed,nb_batch,batch_size_sample,nb_ech,batch_size,batch_eval,num_epochs,ttt,selection):
    
    agent = training_agent(Xpool,Ypool,Xtest,Ytest,Basenet,Evalnet,acquisition_f,optimizer)
    
    agent.evaluate(batch_eval[0])
    
    for k in range(nb_batch):
        
        agent.pick(batch_size_sample[k],nb_ech,batch_size[k],num_epochs[k])
        agent.evaluate(batch_eval[k])
        print("batch",k,"seed",seed)

    agent.save(ttt,selection,seed)
    
def agent_random(Xpool,Ypool,Xtest,Ytest,Basenet,Evalnet,acquisition_f,optimizer,seed,nb_batch,batch_size_sample,nb_ech,batch_size,batch_eval,num_epochs,ttt):
    
    agent = training_agent(Xpool,Ypool,Xtest,Ytest,Basenet,Evalnet,acquisition_f,optimizer)
    
    agent.evaluate(batch_eval[0])
    
    for k in range(nb_batch):
        
        agent.edit_selection(batch_size_sample[k])
        agent.evaluate(batch_eval[k])
        print("batch",k,"seed",seed)

    agent.save(ttt,seed)
    
    
    
    
'''  
 ------------------------------------------------------------------------------------------------------------------   
'''       
    
class training_agent():
    
    
    def __init__(self, Xpool,Ypool,Xtest,Ytest,BaseNet,EvalNet,acquisition_f,optimizer):
        
        self.size = Xpool.shape[0]
        self.Xselected = [np.random.choice(self.size)]
        self.Xpool = Xpool
        self.Ypool = Ypool
        self.inf_loader = torch.utils.data.DataLoader(csvDataset(Xpool,Ypool,transform=ToTensor()),batch_size=self.size, shuffle=False)
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.EvalNet = EvalNet
        self.BaseNet = BaseNet
        self.acquisition_f = acquisition_f
        self.results = torch.zeros((1000))
        self.optimizer=optimizer
        
    def evaluate(self,batch_size):
        
        selection = self.Xselected
        model = self.EvalNet(dropout_rate=0.1)
        model = model.float().cuda()
        optimizer = optim.Adam(model.parameters(), weight_decay=0)
        criterion = F.binary_cross_entropy_with_logits
        
        
        Xte,Yte = self.Xtest,self.Ytest
        inference_dataset = csvDataset(Xte,Yte,transform= ToTensor())
        inference_loader = torch.utils.data.DataLoader(inference_dataset,batch_size=Xte.shape[0], shuffle=False)
        
        Xtr,Ytr = self.Xpool[selection],self.Ypool[selection]
        file_dataset = csvDataset(Xtr,Ytr,transform= ToTensor())
        final_loader = torch.utils.data.DataLoader(file_dataset,batch_size=np.asscalar(batch_size), shuffle=False)

        model,train_accuracy,ep = train_model_cc_fast(model, final_loader, criterion,optimizer,Xtr.shape[0], num_epochs=50)
        model.eval()
        with torch.no_grad():
            for i in inference_loader:
                inputs = i['data']
                labels = i['label']
                inputs, labels = inputs.cuda(), labels.cuda()
                out = model.forward(inputs)
                pred = (out.cpu().numpy()>0)*1.
                labels = (labels.cpu().numpy())*1.
        
        correct =(np.sum(pred==labels)/Xte.shape[0])
        self.results[len(selection)]= correct
        
        
    def pick(self,batch_size_sample,nb_ech,batch_size,num_epochs):
        
        selection = self.Xselected
        Xtr,Ytr = self.Xpool[selection],self.Ypool[selection]
        file_dataset = csvDataset(Xtr,Ytr,transform= ToTensor())
        final_loader = torch.utils.data.DataLoader(file_dataset,batch_size=np.asscalar(batch_size), shuffle=False)
        
        criterion = F.binary_cross_entropy_with_logits
        
        if self.optimizer =='VOGN':
            model = self.BaseNet()
            model = model.float().cuda()
            op = VOGN(model, train_set_size=1000,prior_prec=30, prec_init=30, num_samples=4)
        else :
            model = self.BaseNet(dropout_rate=0.15)
            model = model.float().cuda() 
            op = optim.Adam(model.parameters(),weight_decay=0)
        model,train_accuracy,ep = train_model_cc_fast(model, final_loader, criterion,op,len(selection), num_epochs)
        print('train',ep)  
        labz=torch.zeros(nb_ech,self.size).cuda()
        predict = torch.zeros(nb_ech,self.size).cuda()

        with torch.no_grad():
            if self.optimizer =='VOGN':
                model.eval()
                for i in range(nb_ech):
                    predictions,lbl = inference_pp(model, self.inf_loader,op,1)
                    predict[i] = predictions.view(self.size)
                    labz[i] = lbl.view(self.size)
            else:
                model.train()
                for i in range(nb_ech):
                    predictions,lbl = inference_(model, self.inf_loader)
                    predict[i] = predictions.view(self.size)
                    labz[i] = lbl.view(self.size)

        predict_train = predict.cpu().numpy()
        self.Xselected = assist(np.argsort(self.acquisition_f(predict_train)),self.Xselected,batch_size_sample)
    
    
        
    def save(self,ttt,selection,seed): 
        print('ok',seed)
        ttt[seed] = self.results
        print(self.Xselected)
        selection += torch.tensor(np.array(self.Xselected)).view(-1,1).float()
        
        
    def edit_selection(self,a): 
        self.Xselected = np.random.choice(self.size,a+len(self.Xselected),replace=False)
        
        
'''  
 ------------------------------------------------------------------------------------------------------------------   
'''
class csvDataset():
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

'''  
 ------------------------------------------------------------------------------------------------------------------   
'''    


def accuracy_bb(model, dataloader,size):
    """ Computes the model's classification accuracy on the train dataset
    Computes classification accuracy and loss(optional) on the test dataset
    The model should return logits
    """
    model.eval()
    with torch.no_grad():
        correct = 0.
        for i in (dataloader):
            inputs = i['data']
            labels = i['label']
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            pred = (outputs>0).float()
            correct += (pred.view(-1,1) == labels).sum().item()
        accuracy = correct / size
        
    return accuracy

'''  
 ------------------------------------------------------------------------------------------------------------------   
'''

def inference_pp(model, data_loader, optimizer,mc_samples):
    for i in (data_loader):
        inputs = i['data']
        labels = i['label']
        inputs, labels = inputs.cuda(), labels.cuda()
    return optimizer.get_mc_predictions(model.forward,inputs,mc_samples=mc_samples)[0],labels



def inference_(model, data_loader):
    for i in (data_loader):
        inputs = i['data']
        labels = i['label']
        inputs, labels = inputs.cuda(), labels.cuda()
    return model.forward(inputs),labels


'''  
 ------------------------------------------------------------------------------------------------------------------   
'''


def assist (b,a,count):
    c = 0
    d = b.shape
    d=9472-1
    while c != count:
        if b[d] not in a:
            a.append(b[d])
            c+=1
        d-=1
    return a
                                                      
'''  
 ------------------------------------------------------------------------------------------------------------------   
'''
              
def train_model_cc_fast(model, trainloader, criterion, optimizer,size, num_epochs=25):

    for epoch in range(num_epochs):
        model.train(True)
        for i in trainloader:
            inputs = i['data']
            labels = i['label']
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

        train_accuracy = accuracy_bb(model, trainloader,size)
        
        if (epoch > 1) & (train_accuracy >0.999):
                break
    return model, train_accuracy,epoch


        