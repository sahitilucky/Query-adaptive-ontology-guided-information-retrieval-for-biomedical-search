import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import random
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import numpy as np
#import torchsample as ts
#from itertools import iter

# Device configuration
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 20
num_classes = 2
batch_size = 100
learning_rate = 0.0001

class trainDataset(Dataset):
    def __init__(self, data_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        train_data = np.load(data_file)
        X = train_data[:,:-1]
        Y = train_data[:,-1]
        X_val_set = X[:1000,:]
        Y_val_set = Y[:1000]
        X=X[1000:,:]
        Y=Y[1000:]
        print ('error here')
        y_ones_idx = np.nonzero(Y)
        trim = y_ones_idx[0][:200000]
        x_ones_trim = X[trim,:]
        y_zeros_idx = np.where(Y==0)
        trim= y_zeros_idx[0][:200000]
        x_zeros_trim = X[trim,:]
        self.inputs = np.concatenate((x_ones_trim, x_zeros_trim), axis=0)
        print (x_ones_trim.shape[0])
        print (x_zeros_trim.shape[0])
        self.labels = np.concatenate((np.array([1]*x_ones_trim.shape[0]), np.array([0]*x_zeros_trim.shape[0])), axis=0)
        self.labels = np.reshape(self.labels, (-1,1))
        print (self.inputs.shape)
        print (self.labels.shape)
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        inputs = Variable(torch.FloatTensor(self.inputs[idx,:]))
        label = self.labels[idx,0]
        return (inputs,label)

class valiDataset(Dataset):
    def __init__(self, data_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        train_data = np.load(data_file)
        X = train_data[:,:-1]
        Y = train_data[:,-1]
        X_val_set = X[:1000,:]
        Y_val_set = Y[:1000]
        self.inputs = X_val_set
        self.labels = np.array(Y_val_set)
        self.labels = np.reshape(self.labels, (-1,1))
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return ( Variable(torch.FloatTensor(self.inputs[idx,:])), Variable(torch.LongTensor(self.labels[idx,:])))

class testDataset(Dataset):
    def __init__(self, data_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        train_data = np.load(data_file)
        X = train_data[:,:]
        self.inputs = X
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return ( Variable(torch.FloatTensor(self.inputs[idx,:])))


# Convolutional neural network (two convolutional layers)
class Feedforward(nn.Module):
    def __init__(self, num_classes=2):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Sequential(
                  nn.Linear(8329,1000),
                  nn.ReLU(),
                  )
        self.fc2 = nn.Sequential(
                  nn.Linear(1000,500),
                  nn.ReLU(),
                  )
        self.linear = nn.Linear(500, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out =self.fc2(out)
        out = self.linear(out)
        return out


# Convolutional neural network (two convolutional layers)
class Disgnet_Feedforward(nn.Module):
    def __init__(self, num_classes=2):
        super(Disgnet_Feedforward, self).__init__()
        self.fc1 = nn.Sequential(
                  nn.Linear(8202,1000),
                  nn.ReLU(),
                  )
        self.fc2 = nn.Sequential(
                  nn.Linear(1000,500),
                  nn.ReLU(),
                  )
        self.linear = nn.Linear(500, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out =self.fc2(out)
        out = self.linear(out)
        return out

# train dataset
train_dataset = trainDataset(data_file='../LC_related_data_2/disgenet_train_data_glove_new_features.npy', transform=transforms.ToTensor())

# train dataset
vali_dataset = valiDataset(data_file='../LC_related_data_2/disgenet_train_data_glove_new_features.npy', transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=vali_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

#model = Feedforward(num_classes).to(device)
model = Disgnet_Feedforward(num_classes).to(device)
#model = torch.load('../LC_related_data_2/Pytorch_models/disgenet_glove_nf.model')
'''
for i, (images, labels) in enumerate(train_loader):
  images = images.to(device)
  print (labels.type())
  labels = labels.to(device)
  print (labels.type())
  outputs = model(images)
  print (outputs.size())
  break
for i, (images, labels) in enumerate(test_loader):
  images = images.to(device)
  print (labels.type())
  labels = labels.to(device)
  print (labels.type())
  outputs = model(images)
  print (outputs.size())
  break
'''
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    #if(epoch == 40):
    #    learning_rate = 0.00001
    #    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i, (queries, targets) in enumerate(train_loader):
        #print (targets.type())
        queries = queries.to(device)
        targets = targets.to(device)
        #print (targets.type())
        # Forward pass
        outputs = model(queries)
        loss = criterion(outputs, targets)
        #print (loss.item())
        #print (loss)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            torch.save(model,'../LC_related_data_2/Pytorch_models/disgenet_temp_glove_nf.model')


#model = torch.load('../LC_related_data_2/Pytorch_models/disgenet_temp.model')
# Test the Classification model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for queries, labels in train_loader:
        queries = queries.to(device)
        labels = labels.to(device)
        outputs = model(queries)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Train Accuracy: {} %'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0
    for queries, labels in test_loader:
        queries = queries.to(device)
        labels = labels.to(device)
        outputs = model(queries)

        _, predicted = torch.max(outputs.data, 1)
        labels = labels.view((labels.size(0)))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))

print ('Saving.....')
# Save the model checkpoint
torch.save(model.state_dict(), '../LC_related_data_2/Pytorch_models/disgenet_glove_nf.ckpt')
torch.save(model,'../LC_related_data_2/Pytorch_models/disgenet_glove_nf.model')

'''
# train dataset
testdataset = testDataset(data_file='../LC_related_data_2/disgenet_trec_test_data_biomed_8k.npy', transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=testdataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
model = torch.load('../LC_related_data_2/Pytorch_models/disgenet_glove_nf.model')
model.eval()
i = 0
model_outputs = np.zeros((testdataset.__len__(),2))
with torch.no_grad():
    for queries in test_loader:
        queries = queries.to(device)
        outputs = model(queries)
        outputs = outputs.cpu().numpy()
        model_outputs[i:i+outputs.shape[0],:] = outputs
        i = i +outputs.shape[0] 
print (model_outputs.shape)
np.save('../LC_related_data_2/Pytorch_models/disgenet_trec_data_ep_glove_nf.npy' , model_outputs)
#    print('Test Accuracy: {} %'.format(100 * correct / total))


# train dataset
testdataset = testDataset(data_file='../LC_related_data_2/disgenet_test_data_biomed_8k.npy', transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=testdataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
model = torch.load('../LC_related_data_2/Pytorch_models/disgenet_glove_nf.model')
model.eval()
i = 0
model_outputs = np.zeros((testdataset.__len__(),2))
with torch.no_grad():
    for queries in test_loader:
        queries = queries.to(device)
        outputs = model(queries)
        outputs = outputs.cpu().numpy()
        model_outputs[i:i+outputs.shape[0],:] = outputs
        i = i +outputs.shape[0] 
print (model_outputs.shape)
np.save('../LC_related_data_2/Pytorch_models/disgenet_ep_glove_nf.npy' , model_outputs)
'''