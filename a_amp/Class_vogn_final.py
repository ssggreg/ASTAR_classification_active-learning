import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from vogn import VOGN
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from scipy import stats

from active_function import *


def agent_cell(Xpool,Ypool,Xtest,Ytest,Basenet,Evalnet,uncertainty,function,function_sec,recalibration,seed,nb_batch,batch_size_sample,nb_ech,batch_size,num_epochs,ttt):
    
    agent = training_agent(Xpool,Ypool,Xtest,Ytest,Basenet,Evalnet,uncertainty,function,function_sec,recalibration)
    
    agent.evaluate()
    
    for k in range(nb_batch):
        
        agent.pick(batch_size_sample[k],nb_ech,batch_size[k],num_epochs[k])
        agent.evaluate()
        print("batch",k,"seed",seed)

    agent.save(ttt,seed)
    
def agent_random(Xpool,Ypool,Xtest,Ytest,Evalnet):
    
    agent = training_agent(Xpool,Ypool,Xtest,Ytest,'',Evalnet,'','','','')
    
    agent.evaluate()
    
    for k in range(nb_batch):
        
        agent.edit_selection(batch_size_sample[k])
        agent.evaluate()
        print("batch",k,"seed",seed)

    agent.save(ttt,seed)
    
    
    
    
'''  
 ------------------------------------------------------------------------------------------------------------------   
'''       
    
class training_agent():
    
    
    def __init__(self,Xpool,Ypool,Xtest,Ytest,Basenet,Evalnet,uncertainty,function,function_sec,recalibration):
        
        self.size = Xpool.shape[0]
        self.Xselected = [np.random.choice(self.size)]
        self.Xpool = Xpool
        self.Ypool = Ypool
        self.inf_loader = torch.utils.data.DataLoader(csvDataset(Xpool,Ypool,transform=ToTensor()),batch_size=self.size, shuffle=False)
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.EvalNet = Evalnet
        self.BaseNet = Basenet
        self.uncertainty = uncertainty
        self.function = function
        self.function_sec = function_sec
        self.recalibration = recalibration
        
        self.results = torch.zeros((1000))
        
        
        
    def evaluate(self):
        
        selection = self.Xselected
        model = self.EvalNet(dropout_rate=0.1)
        model = model.float().cuda()
        optimizer = optim.Adam(model.parameters(), weight_decay=0)
        criterion = F.binary_cross_entropy_with_logits
        
        batch_size = int(1 + np.floor(len(selection)/30))
        
        Xte,Yte = self.Xtest,self.Ytest
        inference_dataset = csvDataset(Xte,Yte,transform= ToTensor())
        inference_loader = torch.utils.data.DataLoader(inference_dataset,batch_size=Xte.shape[0], shuffle=False)
        
        Xtr,Ytr = self.Xpool[selection],self.Ypool[selection]
        file_dataset = csvDataset(Xtr,Ytr,transform= ToTensor())
        final_loader = torch.utils.data.DataLoader(file_dataset,batch_size=batch_size, shuffle=False)

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
        
        
        if self.uncertainty == 'mc_dropout':
            
            selection = self.Xselected
            Xtr,Ytr = self.Xpool[selection],self.Ypool[selection]
            file_dataset = csvDataset(Xtr,Ytr,transform= ToTensor())
            final_loader = torch.utils.data.DataLoader(file_dataset,batch_size=np.asscalar(batch_size), shuffle=False)

            criterion = F.binary_cross_entropy_with_logits
            
            model = self.BaseNet(dropout_rate=0.15)
            model = model.float().cuda() 
            op = optim.Adam(model.parameters(),weight_decay=0)
            
            
            model,train_accuracy,ep = train_model_cc_fast(model, final_loader, criterion,op,len(selection), num_epochs)
            labz=torch.zeros(nb_ech,self.size).cuda()
            predict = torch.zeros(nb_ech,self.size).cuda()
                        
            with torch.no_grad():  
                model.train()
                for i in range(nb_ech):
                    predictions,lbl = inference_(model, self.inf_loader)
                    predict[i] = predictions.view(self.size)
                    labz[i] = lbl.view(self.size)
                    
            predict_train = predict.cpu().numpy()        
            
            
        elif self.uncertainty == 'bootstrap':
            
            selection = self.Xselected
            
            criterion = F.binary_cross_entropy_with_logits
        
            predict_train = bootstrap(nb_ech,self.BaseNet,criterion,self.Xpool,self.Ypool,self.Xselected,self.size)
            
            predict_train = predict_train.reshape(nb_ech,-1)
            
            
        elif self.uncertainty == 'vogn':
        
            selection = self.Xselected
            Xtr,Ytr = self.Xpool[selection],self.Ypool[selection]
            file_dataset = csvDataset(Xtr,Ytr,transform= ToTensor())
            final_loader = torch.utils.data.DataLoader(file_dataset,batch_size=np.asscalar(batch_size), shuffle=False)

            criterion = F.binary_cross_entropy_with_logits

            model = self.BaseNet()
            model = model.float().cuda()
            op = VOGN(model, train_set_size=1000,prior_prec=30, prec_init=30, num_samples=4)

            model,train_accuracy,ep = train_model_cc_fast(model, final_loader, criterion,op,len(selection), num_epochs)
            labz=torch.zeros(nb_ech,self.size).cuda()
            predict = torch.zeros(nb_ech,self.size).cuda()

            with torch.no_grad():
                model.eval()
                for i in range(nb_ech):
                    predictions,lbl = inference_pp(model, self.inf_loader,op,1)
                    predict[i] = predictions.view(self.size)
                    labz[i] = lbl.view(self.size)


            predict_train = predict.cpu().numpy()
            
            
            
        if self.recalibration =='yes':

            predict_train = recalibrate(predict_train,self.Ypool,selection)
            
            
        if callable(self.function):

            valued_pool = self.function(predict_train)

            valued_pool[self.Xselected]=0

            sorted_pool = np.argsort(valued_pool)


            if self.function_sec =='ebmal':

                k = 2

                pre_selec = assist(sorted_pool,[],5*k,self.size)

                kmeans = KMeans(n_clusters=k, random_state=0).fit(self.Xpool[pre_selec])

                for i in range(k):


                    e = -euclidean_distances(self.Xpool[pre_selec],kmeans.cluster_centers_[i].reshape(1, -1))

                    s = np.argsort(e.reshape(-1))                              

                    g = np.array([pre_selec[i] for i in s])

                    self.Xselected = assist(g,self.Xselected,1,len(g))

            else :

                self.Xselected = assist(sorted_pool,self.Xselected,batch_size_sample,self.size)
    
        print(len(self.Xselected))
        
        
        
        
    def save(self,ttt,seed): 
        print('ok',seed)
        ttt[seed] = self.results
        
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


def assist (b,a,count,size):
    c = 0
    d = b.shape
    d=size-1
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
        
    return model, train_accuracy,epoch

                                                      
'''  
 ------------------------------------------------------------------------------------------------------------------   
'''

def recalibrate(predz,Ypool,selection):
    
    p=np.arange(0,1.1,0.1)
    
    true_labels = Ypool[selection]
    
    predictions = predz[:,selection]
    
    nech,poolsize = predz.shape
        
    rec_dataset = make_rec_dataset(predictions,true_labels)
        
    iso_reg = IsotonicRegression().fit(rec_dataset[:,0],rec_dataset[:,1]).predict(p)
    
    
    new_ypool_prediction = np.zeros(predz.shape)
    
            
    for i in range(poolsize):
        
        data_to_d = predz[:,i]
        est_quant_ = np.quantile(data_to_d,p)
        iso_reg[0]=0
        iso_reg[-1]=1
        new_distribution = IsotonicRegression().fit(iso_reg,est_quant_)
        
        r = np.random.rand(nech)
        new_ypool_prediction[:,i]= new_distribution.predict(r)
        
        #print(np.isnan(np.sum(new_ypool_prediction[:,i])),'pool')
        
    return new_ypool_prediction


def make_rec_dataset(predictions,true_labels):
    
    p = np.arange(0,1.01,0.01)
    
    l,_ = true_labels.shape
        
    dataset = np.zeros((l+2,2))
    dataset[l]+=1
    
    emp_quant = empirical_quantile(predictions,true_labels,p)
    iso_reg = IsotonicRegression().fit(p,emp_quant)
    
    for i in range(l):
        dataset[i,0] = stats.percentileofscore(predictions[:,i], true_labels[i])/100
        dataset[i,1] = iso_reg.predict(np.array([dataset[i,0]]))
    
    return dataset
    
def empirical_quantile(v,t,p):
    true_quantiles = np.zeros(p.shape[0])
    l = t.shape[0]

    for i in range(l):
        true_quantiles+= quantile_v2(v[:,i],t[i],p)
        
    return true_quantiles/l

def quantile_v2(v,t,p):
    
        
    quantiles_bins = np.quantile(v,p)
    
    true_quantiles = t<quantiles_bins
    
    return true_quantiles*1


'''  
 ------------------------------------------------------------------------------------------------------------------   
'''


def recalibrate_b(predz,Ypool,selection):
    
    
    predictions = predz[:,selection]
    
    nech,poolsize = predz.shape
        
    new_ypool_prediction = np.zeros(predz.shape)
    
    
    true_labels = torch.from_numpy(Ypool[selection]).view(-1).float()
    stds = torch.from_numpy(np.std(predictions,axis=0))
    means = torch.from_numpy(np.mean(predictions,axis=0))
    
    s = torch.tensor(1.5,requires_grad=True)
    
    optimizer = optim.Adam([s], weight_decay=0)
    

    for i in range(1000): 
        
        l = np.random.choice(len(selection),1, replace=False)
                
        optimizer.zero_grad()
                
        t = (stds**2)/torch.sqrt(stds*s)
        p = true_labels-means
        p = (p**2)/2
        p = p/s**2
        p = torch.exp(p)/t                    
        p = torch.log(p)
        loss = torch.mean(-p[l])                    
        
        loss.backward()
            
        optimizer.step()
        
    print(s,'s')
    
    s = s.detach().numpy()
            
    new_variance = np.std(predz,axis=0)*s
        
    return new_variance,s
        
        
def ence(predz,Ypool,selection,bins):
      
    true_labels = Ypool[selection]

    predictions = predz[:,selection]

    nech,size = predictions.shape

    e_nce = 0


    for i in range(bins):
        
        selection_bin = [i*int(size/bins),(i+1)*int(size/bins)-1]
        
        m_var = np.sqrt(np.mean(np.std(predictions[:,selection_bin],axis=0)**2))
        
        emse = np.sqrt(np.mean((true_labels[selection_bin]-np.mean(predictions[:,selection_bin],axis=0))**2))

        e_nce+= abs(m_var-emse)/(m_var+1e-10)

    return e_nce/bins 
                                                      
'''  
 ------------------------------------------------------------------------------------------------------------------   
'''



def bootstrap(nb_ech,BaseNet,criterion,Xpool,Ypool,Xselected,size):
    
    l = len(Xselected)
    predz = torch.zeros((nb_ech,Ypool.shape[1],size)).float().cuda()
    
    Xte,Yte = Xpool,Ypool
    inference_dataset = csvDataset(Xte,Yte,transform= ToTensor())
    inference_loader = torch.utils.data.DataLoader(inference_dataset,batch_size=size, shuffle=False)
    
    for i in range(nb_ech):
            
        xselection = np.random.choice(l,l,replace=True) 
        selection =[Xselected[j] for j in xselection]
        
        model = BaseNet(dropout_rate=0.1)
        model = model.float().cuda()
        optimizer = optim.Adam(model.parameters(), weight_decay=0)        
        
        
        Xtr,Ytr = Xpool[selection],Ypool[selection]
        file_dataset = csvDataset(Xtr,Ytr,transform= ToTensor())
        final_loader = torch.utils.data.DataLoader(file_dataset,batch_size=3, shuffle=False)

        model,error,ep = train_model_cc_fast(model, final_loader, criterion,optimizer,Xtr.shape[0], num_epochs=50)
        model.eval()
        predictions,lbl = inference_(model, inference_loader)
        predz[i] = predictions.transpose(0, 1)
        
    
    return predz.detach().cpu().numpy()                              