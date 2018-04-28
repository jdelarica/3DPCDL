# Audiovisual Systems Engineering Final Thesis 
# 3D Point Cloud Correspondences using Deep Learning
#
# Barcelona School of Telecommunications Engineering 
# Universitat Politècnica de Catalunya
#
# test.py computes the test of the network selected, providing the average testing and accuracy values.
#
# Author: Javier de la Rica
# Date: February 2018

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
dropout = True
color = True
outdir = '/imatge/jdelarica/work/PycharmProjects/TFG/tests_512_100_50_1/sgd_lr0.0001_mom0.0_wd0.0_drop_color'

print('Loading data ... ', end='')
dataset = datasets.FreiburgRGBDDataset('data',mode='test',color=color)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
print('OK.')

print('Creating network ... ', end='')
net = networks.NetFC(nodes=[dataset.get_input_size(),600,100,1], dropout=dropout)
net.load_state_dict(torch.load(os.path.join(outdir,'net.pkl')))
print('OK.')

print('Creating loss and optimizer ... ', end='')
criterion = nn.BCELoss(size_average=False)
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
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

print('Starting test ... ', end='')

net.train(False)

# Accumulate loss and accuracy batch by batch
loss_total = 0.0
accu_total = 0.0

# go to all batches
for i, data in enumerate(dataloader, 0):

    # Get inputs and go to GPU 
    inputs, labels = data
    if cuda:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    # zero the parameter gradients
    #optimizer.zero_grad()

    # forward pass
    outputs = net(inputs)
    loss = criterion(outputs, labels)
            
    # store statistics
    loss_total += loss.data[0]        
    accu_total += torch.sum((outputs.data>0.5).float() == labels.data)

# Compute losses and accuracies
l = loss_total / len(dataset)
a = accu_total / len(dataset)
    
print('test_loss: {:.4f} test_accu: {:.4f} '.format(l, a))

