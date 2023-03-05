import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
torch.cuda.set_device(0)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 100
batch_size = 1024
num_classes = 10

# Modify the address according to your own environment
address = '/home/'


train_dataset = datasets.FashionMNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.FashionMNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break



class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=8,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1) # (1(28-1) - 28 + 3) / 2 = 1

        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0) # (2(14-1) - 28 + 2) = 0                                       
        # 14x14x8 => 14x14x16
        self.conv_2 = torch.nn.Conv2d(in_channels=8,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1) # (1(14-1) - 14 + 3) / 2 = 1                 
        # 14x14x16 => 7x7x16   
        
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0)

        self.linear_1 = torch.nn.Linear(7*7*16, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0.0, 0.1)
                m.bias.data.zero_()
                if m.bias is not None:
                    m.bias.detach().zero_()
      
    def forward(self, x):
        
        out = self.conv_1(x)
        #Calculate the energy
        out_1=out
        out = F.relu(out)
        out_1_energy = torch.mul(out_1, out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out_2 = out
        out = F.relu(out)
        out_2_energy = torch.mul(out_2, out)
        out = self.pool_2(out)
        
        logits = self.linear_1(out.view(-1, 7*7*16))
        probas = F.softmax(logits, dim=1)
        return logits, probas,out_1_energy,out_2_energy
torch.manual_seed(random_seed)
model = ConvNet(num_classes=num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

address1 = address+'train_reslut.txt'
address2 = address+'train_loss.txt'
energy_layer_1 = [0]*8
energy_layer_2 = [0]*16
for i in range(8) :
    energy_layer_1[i] =open(address+'energy_layer_1_{}.txt'.format(i+1),'w',encoding = 'utf-8')
for i in range(16) :
    energy_layer_2[i] =open(address+'energy_layer_2_{}.txt'.format(i+1),'w',encoding = 'utf-8')


file1 = open(address1,'w', encoding = 'utf-8')
file2 = open(address2,'w', encoding = 'utf-8')

file1.writelines("Learning_Rate:"+ str(learning_rate)+'\n'
                +"Batch_Size:"+str(batch_size)+'\n'
                +"Num_Epochs:"+str(num_epochs)+'\n'
                +"num_classes:"+str(num_classes)+'\n'
                +"optimizer:" + str(optimizer) + '\n'
                                     
                )

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)[:2]
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    

start_time = time.time()    
for epoch in range(num_epochs):
    
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(device)
        targets = targets.to(device)

        ### FORWARD AND BACK PROP
        logits, probas,layer1_energy,layer2_energy = model(features)
        layer1_energy = layer1_energy.cpu().detach().numpy().mean(0)
        layer2_energy = layer2_energy.cpu().detach().numpy().mean(0)
        layer1_energy = layer1_energy.reshape(8,784)
        layer2_energy = layer2_energy.reshape(16,196)
        layer1_energy = layer1_energy.tolist()
        layer2_energy = layer2_energy.tolist()
           
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader),cost))
            file1.writelines('Epoch: %03d/%03d | Batch %03d/%03d| Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx,len(train_loader), cost)+'\n')
            for i in range(8):
                energy_layer_1[i].writelines(str([round(p,2) for p in layer1_energy[i]])+'\n')
            for j in range(16):
                energy_layer_2[j].writelines(str([round(q,2) for q in layer2_energy[j]])+'\n')
        
        
    model = model.eval()
    print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
          epoch+1, num_epochs, 
          compute_accuracy(model, train_loader)))
    file1.writelines('Epoch: %03d/%03d training accuracy: %.2f%%' % (
          epoch+1, num_epochs, 
          compute_accuracy(model, train_loader))+'\n')
    file2.writelines('%.2f' % compute_accuracy(model, train_loader).item() + '\n')

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    file1.writelines('Time elapsed: %.2f min' % ((time.time() - start_time)/60)+'\n')
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
file1.writelines('Total Training Time: %.2f min' % ((time.time() - start_time)/60)+'\n')

with torch.set_grad_enabled(False): 
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))
    file1.writelines('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader))+'\n')

file1.close()
file2.close()
for i in range(8) :
    energy_layer_1[i].close()
for i in range(16) :
    energy_layer_2[i].close()
