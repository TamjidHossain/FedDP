# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 01:16:46 2021

@author: tamjid
"""



#%%
import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from numpy import linalg as LA

from rdp_accountant import compute_rdp  # pylint: disable=g-import-not-at-top
from rdp_accountant import get_privacy_spent
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
#%%

df = pd.read_csv('D:/household_power_consumption/processed_household_power_consumption.csv')


df_train = df[df.years == 2006].sample(frac = 0.8).append(df[df.years == 2007].sample(frac = 0.8)).append(
    df[df.years == 2008].sample(frac = 0.8)).append(df[df.years == 2009].sample(frac = 0.8)).append(df[df.years == 2010].sample(frac = 0.8))


df_test = pd.concat([df,df_train]).drop_duplicates(keep=False)

x_train = df_train[df_train.columns[:-1]].copy().values
y_train = df_train[df_train.columns[-1:]].copy().values

x_test = df_test[df_test.columns[:-1]].copy().values
y_test = df_test[df_test.columns[-1:]].copy().values

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

#%%
class Parser:
    def __init__(self):
        self.epochs = 100
        self.lr = 0.001
        self.test_batch_size = 8
        self.batch_size = 24
        self.log_interval = 10
        self.seed = 1
    
args = Parser()
torch.manual_seed(args.seed)

mean = x_train.mean(0, keepdim=True)
dev = x_train.std(0, keepdim=True)
mean[:, 3] = 0.
dev[:, 3] = 1.
x_train = (x_train - mean) / dev
x_test = (x_test - mean) / dev
train = TensorDataset(x_train, y_train)
test = TensorDataset(x_test, y_test)
# train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
# test_loader = DataLoader(test, batch_size=args.test_batch_size, shuffle=True)
#%%

class t_model(nn.Module):
    def __init__(self):
        super(t_model, self).__init__()
        self.fc1 = nn.Linear(24, 32)
        self.fc2 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = x.view(-1, 24)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        x = self.fc3(x)
        return x
    
#Return the samples that each client is going to have as a private training data set. This is a not overlapping set
def get_samples(num_clients, train_len):
    # tam = len(mnist_trainset)
    tam = train_len
    split= int(tam/num_clients)
    split_ini = split
    indices = list(range(tam))
    init=0
    samples = []
    for i in range(num_clients):     
        t_idx = indices[init:split]
        t_sampler = SubsetRandomSampler(t_idx)
        samples.append(t_sampler)
        init = split
        split = split+split_ini
    return samples
#%%

def uniform_proposal(x, delta=2.0):
    return np.random.uniform(x - delta, x + delta)

def metropolis_sampler(p, nsamples, proposal=uniform_proposal):
    x = 1 # start somewhere

    for i in range(nsamples):
        trial = proposal(x) # random neighbour from the proposal distribution
        acceptance = p(trial)/p(x)

        # accept the move conditionally
        if np.random.uniform() < acceptance:
            x = trial
        yield x
#%%


class server():
    def __init__(self, train, test, train_len, test_len, number_clients, p_budget, epsilon, sigmat = 1.12):
        self.model = t_model()
        self.sigmat = sigmat   
        self.n_clients = number_clients
        self.samples = get_samples(self.n_clients, train_len)
        self.clients = list()
        for i in range(number_clients):
            # loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=24, sampler=self.samples[i])
            loader = DataLoader(train, batch_size=24, sampler=self.samples[i])
            self.clients.append(client(i, loader, self.model.state_dict()))
        self.p_budget = p_budget
        self.epsilon = epsilon
        self.testLoader = torch.utils.data.DataLoader(test, batch_size=24)
        self.device = torch.device("cuda:0""cuda:0" if torch.cuda.is_available() else "cpu")
        self.orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                list(range(5, 64)) + [128, 256, 512])
        
        
    #Evaluates the accuracy of the current model with the test data.  
    def eval_acc(self):
        test_lossList = []
        self.model.to(self.device)
        # running_loss = 0
        # accuracy = 0
        self.model.eval()
        test_loss = 0
        for data, target in self.testLoader:
            output = self.model(data)
            test_loss += F.mse_loss(output.view(-1), target, reduction='sum').item()
            predection = output.data.max(1, keepdim=True)[1]
            
        test_loss /= len(self.testLoader.dataset)
        print('Test set: Average loss: {:.4f}'.format(test_loss))
        return test_loss
        
        # suma=0
        # total = 0
        # running_loss = 0
        # for images, labels in self.testLoader:            
        #     images, labels = images.to(self.device), labels.to(self.device) 
        #     output = self.model.forward(images)             
        #     ps = torch.exp(output)
        #     top_p, top_class = ps.topk(1, dim=1)
        #     equals = top_class == labels.view(*top_class.shape)
        #     total += equals.size(0)
        #     suma = suma + equals.sum().item()
        # else:      
        #     print('Accuracy: ',suma/float(total))
    
    def sanitaze(self,mt, deltas, norms, sigma, state_dict):    
        new_dict = {}
        for key, value in state_dict.items():
            S=[]
            for i in range(len(norms)):        
                S.append(norms[i][key])
            S_value = np.median(S)      
            wt = value
            prom = 1/float(mt)       
            suma = 0
            for i in range(len(deltas)):    
                clip = (max(1, float(norms[i][key]/S_value)))            
                suma = suma + ((deltas[i][key] / clip ))
            # noise = np.random.normal(0, float(S_value * sigma), size = suma.shape)
            # print(suma.shape)
            # noise = 0
            if (len(suma.shape)==2):
                u,v = suma.shape
                mu = 0
                sigma = float(S_value * sigma)
                gamma = 0.1
                # p = lambda x: 1./sigma/np.sqrt(2*np.pi)*np.exp(-((x-mu)**2)/2./sigma/sigma)
                p = lambda x: 1./sigma/np.sqrt(2*np.pi)*np.exp(-((x-mu - np.sqrt(2*gamma)*sigma)**2)/2./sigma/sigma)
                samples1 = list(metropolis_sampler(p, u*v))
                sample1 = np.array(samples1)
                noise1 = sample1.reshape((u,v))
            else:
                u = len(suma)
                mu = 0
                sigma = float(S_value * sigma)
                gamma = 0.1
                # p = lambda x: 1./sigma/np.sqrt(2*np.pi)*np.exp(-((x-mu)**2)/2./sigma/sigma)
                p = lambda x: 1./sigma/np.sqrt(2*np.pi)*np.exp(-((x-mu - np.sqrt(2*gamma)*sigma)**2)/2./sigma/sigma)
                samples2 = list(metropolis_sampler(p, u))
                noise1 = np.array(samples2)
                
            noise = noise1

            suma = suma.cpu().numpy()
            suma = suma*prom
            noise = noise*prom
            suma = suma + noise 

            suma = torch.from_numpy(suma)
            suma = wt + suma.float()
            new_dict[key] = suma
            
        return new_dict
            
            
    def server_exec(self,mt):    
        i=1
        lossList = []
        loss_valuesListAllFinal = []
        while(True):
#             clear_output()
            print('Comunication round: ', i)
            lossList.append(self.eval_acc())        
            rdp = compute_rdp(float(mt/len(self.clients)), self.sigmat, i, self.orders)
            _,delta_spent, opt_order = get_privacy_spent(self.orders, rdp, target_eps=self.epsilon)
            print('Delta spent: ', delta_spent)
            print('Delta budget: ', self.p_budget)  
            
            if self.p_budget < delta_spent:
                break
            
            Zt = np.random.choice(self.clients, mt)      
            deltas = []
            norms = []
            # print('hello')
            loss_valuesListAll = []
            for client in Zt:
                # print(client)
                deltaW, normW, loss_valuesList = client.update(self.model.state_dict())   
                deltas.append(deltaW)
                norms.append(normW)  
                loss_valuesListAll.append(loss_valuesList)
            loss_valuesListAllFinal.append(loss_valuesListAll)
            self.model.to('cpu')
            new_state_dict = self.sanitaze(mt, deltas, norms, self.sigmat, self.model.state_dict())
            self.model.load_state_dict(new_state_dict)
            i+=1
        return self.model, lossList, loss_valuesListAllFinal
#%%

class client():
    def __init__(self, number, loader, state_dict, batch_size = 24, epochs=10, lr=0.001):
        self.number = number
        self.model = t_model()
        self.model.load_state_dict(state_dict)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.device =  device =  torch.device("cuda:0""cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataLoader = loader                                       
                                           
    def update(self, state_dict):
        
        w0 = state_dict
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        running_loss = 0
        accuracy = 0
        
        loss_valuesList = []
        for e in range(self.epochs):
            # Model in training mode, dropout is on
            self.model.train()
            accuracy=0
            running_loss = 0
            loss_values = 0
            for data, target in self.dataLoader:  
                # print('data', data )
                # print('target', target)
                data, target = data.to(self.device), target.to(self.device)                       
                self.optimizer.zero_grad()
                
                prediction = self.model.forward(data)
                # print(self.model)
                # print(prediction)
                # prediction = self.model(output)
                # # # loss = self.criterion(output, labels)
                # print('pred: ',prediction.view(-1))
                # print('tar: ',target)
                loss = F.mse_loss(prediction.view(-1), target)
                loss.backward()
                self.optimizer.step()            
                running_loss += loss.item() 
                loss_values = running_loss / len(self.dataLoader)
        loss_valuesList.append(loss_values)
        S ={} 
        wt1 = {}
        for key, value in w0.items():
            wt1[key] = self.model.state_dict()[key]  - value   
            S[key] = LA.norm(wt1[key].cpu(), 2)
        return wt1, S, loss_valuesList

#%%
num_clients = 100
train_len = len(x_train)
test_len = len(x_test)
serv = server(train, test, train_len, test_len, num_clients, 0.001, 8)
#%%


# serv = server(num_clients, 0.001, 8)
model, lossList, loss_valuesListAllFinal = serv.server_exec(30)
plt.plot(lossList)
#%%


















