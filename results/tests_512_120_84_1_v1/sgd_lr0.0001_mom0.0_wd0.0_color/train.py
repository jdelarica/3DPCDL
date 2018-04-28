from __future__ import print_function
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np
import shutil
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import datasets
import networks

# Variables
nepochs = 500
lr = 0.0001
momentum = 0.0
weight_decay = 0.0
dropout = False
color = True
outdir = 'tests/sgd_lr0.0001_mom0.0_wd0.0_color'

# Copy ourselves to outdir to replicate results
if not os.path.exists(outdir):
    os.makedirs(outdir)
shutil.copy(__file__, os.path.join(outdir,'train.py'))

print('Loading data ... ', end='')
rotate = datasets.RandomRotate(p=1.0)
dataset = {}
dataloader = {}
dataset['train'] = datasets.FreiburgRGBDDataset('data',mode='train',color=color, transform=None, seed=42)
dataloader['train'] = DataLoader(dataset['train'], batch_size=16, shuffle=True, num_workers=0)
dataset['val'] = datasets.FreiburgRGBDDataset('data',mode='val',color=color, seed=42)
dataloader['val'] = DataLoader(dataset['val'], batch_size=16, shuffle=True, num_workers=0)
print('OK.')

#print (len(dataset['train']))
#print (len(dataloader['train']))

print('Creating network ... ', end='')
net = networks.NetFC(nodes=[dataset['train'].get_input_size(),600,100,1], dropout=dropout)
print('OK.')

print('Creating loss and optimizer ... ', end='')
criterion = nn.BCELoss(size_average=False)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
print('OK.')

cuda = torch.cuda.is_available()
if not cuda:
    print ("No GPU found!!!!!!!!!!!!")

if cuda:
    print ('Use GPU ... ', end='')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs! ", end='')
        net = nn.DataParallel(net)
    net.cuda()
    criterion.cuda()
    print('OK.')

print('Starting Training:')
print ('')

# Variables to keep history of losses and accuracies
loss_epochs = {'train': [], 'val': []}
accu_epochs = {'train': [], 'val': []}

# Store best accuracy to save model
best_acc = 0.0
best_epoch = -1

# Empty file with data
open(os.path.join(outdir,'train.txt'),'w').close()

for epoch in range(nepochs):  # loop over the dataset multiple times

    print('Epoch {}/{}: '.format(epoch+1, nepochs), end='')
    
    # for each epoch, we first train the network and then evaluate the validation data
    for phase in ['train','val']:

        # Set model to training or not
        if phase == 'train':
            net.train(True)  # Set model to training mode
        else:
            net.train(False)  # Set model to evaluate mode

        # Accumulate loss and accuracy batch by batch
        loss_total = 0.0
        accu_total = 0.0

        # go to all batches
        for i, data in enumerate(dataloader[phase], 0):

            # Get inputs and go to GPU 
            inputs, labels = data
            if cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # backward + optimize
            if phase == 'train':
                loss.backward()
                optimizer.step()
                
            # store statistics
            loss_total += loss.data[0]        
            accu_total += torch.sum((outputs.data>0.5).float() == labels.data)

        # Compute losses and accuracies
        l = loss_total / len(dataset[phase])
        a = accu_total / len(dataset[phase])
    
        # Append loss and accuracy to vector
        loss_epochs[phase].append(l)
        accu_epochs[phase].append(a)

        print('{}_loss: {:.4f} {}_accu: {:.4f} '.format(phase, l, phase, a), end='')

        if phase=='val':
            if a > best_acc:
                best_acc = a
                best_epoch = epoch + 1
                torch.save(net.state_dict(), os.path.join(outdir,'net.pkl'))

    # Save current loss and accuracy epoch by epoch in file
    with open(os.path.join(outdir,'train.txt'),'a') as f:
        f.write('{:03d} {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(epoch+1,
                                                              loss_epochs['train'][epoch],
                                                              accu_epochs['train'][epoch],
                                                              loss_epochs['val'][epoch],
                                                              accu_epochs['val'][epoch]))
        f.flush()

    print('')


print('Finished Training')

# Save best epoch for reference
print('Saving data ... ', end='')
with open(os.path.join(outdir,'best.txt'),'w') as f:
    f.write('{:03d}  {:.4f}\n'.format(best_epoch, best_acc))
print('OK')

print('Saving plots ... ', end='')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(range(1,nepochs+1),accu_epochs['train'], label='train')
plt.plot(range(1,nepochs+1),accu_epochs['val'], label='val')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.xlim(1,nepochs+1)
plt.legend()
plt.grid()
plt.savefig(os.path.join(outdir,'train_accu.png'))
plt.figure(2)
plt.plot(range(1,nepochs+1),loss_epochs['train'], label='train')
plt.plot(range(1,nepochs+1),loss_epochs['val'], label='val')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.xlim(1,nepochs+1)
plt.legend()
plt.grid()
plt.savefig(os.path.join(outdir,'train_loss.png'))
print('OK.')
