

import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torchvision.transforms as transforms



input_size = 784
batch_size = 1000
num_epochs = 100
learning_rate = 0.001
hidden_size = 50
number_H =2 

# Modify the address according to your own environment
address = '/home/'+'{}'.format(hidden_size)+'_'+'{}'.format(learning_rate)+'_'+'{}'.format(number_H)
address1 = address+'loss.txt'
address2 = address + 'layers_energy.txt'
address3 = address + 'model_energy.txt'

file1 = open(address1,'w')
file2 = open(address2,'w')
file3 = open(address3,'w')

train_datasets = dsets.MNIST(root = 'Datasets', train = True, download = True, transform = transforms.ToTensor())
test_datasets = dsets.MNIST(root = 'Datasets', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_datasets, batch_size = batch_size, shuffle = False)


class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.ReLU()
        self.hidden = hidden
        self.linearH = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(number_H)])
        self.out = nn.Linear(hidden, num_classes)
        
        
    def forward(self, x):
        tensor = torch.tensor((),dtype = torch.float32)
        Power =tensor.new_zeros(number_H,self.hidden)
        x = self.linear(x)
        x_in = x
        x = self.r(x)
        Power_in=torch.sum(torch.mul(x_in, x),dim=0)
        Power_in = Power_in/batch_size
        
        for i in  range(number_H):
            x = self.linearH[i](x)
            x_linear= x
            x = self.r(x)
            Power[i]=torch.sum(torch.mul(x_linear, x),dim=0)/batch_size
            
        out = self.out(x)
        Power_out=torch.sum(torch.mul(out,out),dim=0)/batch_size
        
        return out, Power_in, Power,Power_out
        



if torch.cuda.is_available():
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10).cuda()
else:
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()

Power_layer_collect=[]
Power_in_collect=[]
Power_hidden_collect=[]
Power_out_collect=[]
Power_model_collect = []
loss_collect = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28*28)).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
        outputs, Power_in, Power_hidden,Power_out= model(images)

        Power_in = Power_in.cpu().detach().numpy()
        Power_hidden = Power_hidden.cpu().detach().numpy()
        Power_out = Power_out.cpu().detach().numpy()

        Power_in_list=Power_in.tolist()
        Power_hidden_list = Power_hidden.tolist()
        Power_out_list = Power_out.tolist()

        Power_in_collect.append(Power_in_list)
        Power_hidden_collect.append(Power_hidden_list)
        Power_out_collect.append(Power_out_list)

        Power_in_sum= Power_in.sum()
        Power_hidden_sum = Power_hidden.sum(axis=1)
        Power_out_sum = Power_out.sum()

        Power_layer = [Power_in_sum]+Power_hidden_sum.tolist()+[Power_out_sum]
        Power_layer = [int(i) for i in Power_layer]
        Power_layer_collect.append(Power_layer)
 
        Power_model = Power_in_sum+Power_hidden_sum.sum()+Power_out_sum
        Power_model = int(Power_model)
        Power_model_collect.append(Power_model)

        optimizer.zero_grad() 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_out=round(loss.item(),2)
        file1.writelines(str(loss_out)+'\n')
        file2.writelines(str(Power_layer)+'\n')
        file3.writelines(str(Power_model)+'\n')
        
        
        if (i+1) % 20 == 0:
            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                  (epoch+1, num_epochs, i+1, len(train_datasets)//batch_size,loss.item()))
            print(Power_model)
            
            
file1.close()
file2.close()
file3.close()

