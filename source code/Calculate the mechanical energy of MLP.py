#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import torch
import os
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
torch.cuda.set_device(0)
import torchvision.transforms as transforms


# In[2]:


input_size = 784
batch_size = 512
num_epochs = 200
learning_rate = 0.001
hidden_size = 50
number_H =2


# In[3]:


def seed_torch(seed=42):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
seed_torch()


# In[4]:


address = '/home/sda/luzhixing/MLP/result/energy/'+'{}'.format(batch_size)+'_{}'.format(hidden_size)+'_'+'{}'.format(learning_rate)+'_'+'{}'.format(number_H)
address1 = address+'_loss+energy_average.txt'

file1 = open(address1,'w')


# In[5]:


train_datasets = dsets.MNIST(root = './Datasets', train = True, download = True, transform = transforms.ToTensor())
test_datasets = dsets.MNIST(root = './Datasets', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_datasets, batch_size = batch_size, shuffle = False)


# In[6]:


class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.SiLU()
        self.hidden = hidden
        self.linearH = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(number_H)])
        self.out = nn.Linear(hidden, num_classes)
        
        
    def forward(self, x):
        tensor = torch.tensor((),dtype = torch.float32)
        Hidden_f_z =tensor.new_zeros(number_H,self.hidden)
        Hidden_z =tensor.new_zeros(number_H,self.hidden)
        x = self.linear(x)
        Fst_z = x[0]
        x = self.r(x)
        Fst_f_z = x[0]

        
        for i in  range(number_H):
            x = self.linearH[i](x)
            Hidden_z[i] = x[0]
            x = self.r(x)
            Hidden_f_z[i] = x[0]
            
        out = self.out(x)
        
        return out,Fst_z,Fst_f_z,Hidden_z,Hidden_f_z


# In[7]:


if torch.cuda.is_available():
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10).cuda()
else:
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()


# In[8]:


Power_layer_collect=[]
Power_in_collect=[]
Power_hidden_collect=[]
Power_out_collect=[]
Power_model_collect = []
loss_collect = []


# In[9]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))
def Tanh_softplus(x):
    return np.tanh(np.log(1+np.exp(x)))


# In[10]:


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i,(features, targets) in enumerate(data_loader):
        #features = features.to(device)
        features = Variable(features.view(-1, 28*28)).cuda()
        
        #targets = targets.to(device)
        targets = Variable(targets).cuda()
        probas = model(features)[0]
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


# In[11]:


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28*28)).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
        outputs,Fst_z,Fst_f_z,Hidden_z,Hidden_f_z = model(images)

        Fst_z = Fst_z.cpu().detach().numpy()
        Fst_f_z = Fst_f_z.cpu().detach().numpy()
        Hidden_z = Hidden_z.cpu().detach().numpy()
        Hidden_f_z = Hidden_f_z.cpu().detach().numpy()
        #print(Fst_z)


        for m in range(hidden_size):
            Fst_z[m] = np.log(np.exp(Fst_z[m])+1)
        for m in range(number_H):
            for n in range(hidden_size):
                Hidden_z[m][n] = np.log(np.exp(Hidden_z[m][n])+1)
              

        

        kinetic_energy_Fst = Fst_f_z - Fst_z
        kinetic_energy_Hidden = Hidden_f_z - Hidden_z
        
        Fst_f_z_out = round(-1*(np.mean(Fst_f_z)),6)
        Fst_z_out = round(-1*(np.mean(Fst_z)),6)
        kinetic_energy_Fst_out=round(-1*(np.mean(kinetic_energy_Fst)),6)
        optimizer.zero_grad() 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_out=round(loss.item(),2)
        if (i+1) % 20 == 0:
            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                  (epoch+1, num_epochs, i+1, len(train_datasets)//batch_size,loss.item()))
            print(Fst_f_z_out,Fst_z_out,kinetic_energy_Fst_out)
            
    text_accuracy = round(compute_accuracy(model, test_loader).item(),2)
    print('Text Acc:'+str(text_accuracy))
    file1.writelines(str(loss_out)+'  '+str(text_accuracy)+'  '+str(Fst_f_z_out)+'  '+str(Fst_z_out)+'  '+str(kinetic_energy_Fst_out)+'\n')           
file1.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 
