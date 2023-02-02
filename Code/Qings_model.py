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
import json
#import torchsample as ts
#from itertools import iter

# Device configuration
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 2
batch_size = 100
learning_rate = 0.00001
data_file = '../qing_training/training_data_glove_biomed_dl.txt'
i=0
'''
train_query_ids = [str(x) for x in random.sample(range(1,51), 10)]
train_query_ids_dict = dict(zip(train_query_ids,range(len(train_query_ids))))
print (train_query_ids_dict)
print (train_query_ids)       
with open('../qing_training/Training_queries_for_glove_dl_new.txt','w') as outfile:
  for qid in train_query_ids:
    outfile.write(qid + '\n')
'''
train_query_ids = []
with open('../qing_training/Training_queries_for_glove_dl_new_fold_5.txt','r') as infile:
  for line in infile:
    train_query_ids += [line.strip()]
train_query_ids_dict = dict(zip(train_query_ids,range(len(train_query_ids))))
print (train_query_ids_dict)
print (train_query_ids)
query_positive_docs = {}
query_negative_docs = {}
X_train = []
Y_train = []
X_test = []
Y_test = []
X_train_order = []
X_test_order = []
with open(data_file, 'r') as infile:
    for line in infile:
        query = line.strip().split(',')[0]
        document = line.strip().split(',')[1]
        if line.strip().split(',')[-1] == '1':
          try:
              query_positive_docs[query] += [document]
          except:
              query_positive_docs[query] = [document]
        else:
          try:
              query_negative_docs[query] += [document]
          except:
              query_negative_docs[query] = [document]
        if query in train_query_ids_dict:
            X_train_order += [(query,document)]
            X_train += [[float(x) for x in line.strip().split(',')[2:-2]]]
            Y_train += [int(line.strip().split(',')[-1])]
        else:
            X_test_order += [(query,document)]
            X_test += [[float(x) for x in line.strip().split(',')[2:-2]]]
            Y_test += [int(line.strip().split(',')[-1])]
        i += 1
        if (i%10000)==0:
          print (i)
        #if (i == 200000):
        #  break
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print (X_train.shape)
print (X_test.shape)
#validation_set_indices = random.sample(range(len(X_train_order)), len(X_train_order)/5)
#other_indices = 
#print ('Number of validation samples:' , len(validation_set_indices))
class trainDataset(Dataset):
    def __init__(self,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs = np.zeros((X_train.shape[0], 471),dtype=float)
        self.inputs[:,:-9] = X_train[:,:-209]
        self.inputs[:,-9:] = X_train[:,-9:]
        self.labels = Y_train
        print (self.inputs.shape)
        print (self.labels.shape)
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        inputs = Variable(torch.FloatTensor(self.inputs[idx,:]))
        label = self.labels[idx]
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
        pass
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return ( Variable(torch.FloatTensor(self.inputs[idx,:])), Variable(torch.LongTensor(self.labels[idx,:])))

class testDataset(Dataset):
    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs = np.zeros((X_test.shape[0], 471),dtype=float)
        self.inputs[:,:-9] = X_test[:,:-209]
        self.inputs[:,-9:] = X_test[:,-9:]
        self.labels = Y_test
        print (self.inputs.shape)
        print (self.labels.shape)
    
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        inputs = Variable(torch.FloatTensor(self.inputs[idx,:]))
        label = self.labels[idx]
        return (inputs,label)


# Convolutional neural network (two convolutional layers)
class Feedforward(nn.Module):
    def __init__(self, num_input_features, num_classes=2):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Sequential(
                  nn.Linear(num_input_features,500),
                  nn.ReLU(),
                  )
        '''
        self.fc2 = nn.Sequential(
                  nn.Linear(500,500),
                  nn.ReLU(),
                  )
        '''
        self.linear = nn.Linear(500, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        #out =self.fc2(out)
        out = self.linear(out)
        return out



# train dataset
train_dataset = trainDataset()

# train dataset
test_dataset = testDataset()

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

num_input_features = 471

model = Feedforward(num_input_features,num_classes).to(device)

for i, (images, labels) in enumerate(train_loader):
  images = images.to(device)
  print (labels.type())
  labels = labels.to(device)
  print (labels.type())
  outputs = model(images)
  print (outputs.size())
  break


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
            torch.save(model,'../LC_related_data_2/Pytorch_models/Qing_glove_biomed_model_temp_new_fold5.model')


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
print ('Saving...')
torch.save(model,'../LC_related_data_2/Pytorch_models/Qing_glove_biomed_model_new_fold5.model')

ranked_scores = {}
test_qids = [str(x) for x in range(1,51) if str(x) not in train_query_ids_dict]
for qid in test_qids:   
    ranked_scores[qid] = []
print ('Test_query_ids:',test_qids)
with torch.no_grad():
    i = 0 
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
        outputs = outputs.cpu().numpy()
        scores = outputs[:,1]
        #print (scores.shape)
        for idx, (qid,docid) in enumerate(X_test_order[i:i+outputs.shape[0]]):
            ranked_scores[qid] += [(docid, str(scores[idx]))]
        i = i +outputs.shape[0]
    print('Test Accuracy: {} %'.format(100 * correct / total))

with open('../LC_related_data_2/Pytorch_models/Qing_glove_biomed_model_predictions_new_fold5.json', 'w') as outfile:
    json.dump(ranked_scores,outfile)
