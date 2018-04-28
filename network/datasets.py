# Audiovisual Systems Engineering Final Thesis 
# 3D Point Cloud Correspondences using Deep Learning
#
# Barcelona School of Telecommunications Engineering 
# Universitat Politècnica de Catalunya
#
# datasets.py provides different dataset classes. The FreiburgRGBDDataset class uses both 
# data and labels and provides them to the network depending on the network mode train/test.
#
# Author: Javier de la Rica
# Date: February 2018

from __future__ import print_function
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


def random_matrix():
    """Generate a 3D random rotation matrix.
    Returns:
    np.matrix: A 3D rotation matrix.
    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M


class RandomRotate(object):

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x):
        y = x
        for m in range(2):
            if np.random.random() < self.p:
                R = random_matrix()
                y[m,:,0:3] = np.dot(x[m,:,0:3],R.T)
        return y

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)
        
class RandomScale(object):

    def __init__(self, p=1.0,mscale=0.2):
        self.p = p
        self.mscale = mscale

    def __call__(self, x):
        y = x
        for m in range(2):
            if np.random.random() < self.p:
                s = (self.mscale*np.random.random()) + 1.0
                y[m,:,0:3] = s*x[m,:,0:3]
        return y
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, mscale={1})'.format(self.p, self.mscale)

class RandomIllumination(object):

    def __init__(self, p=1.0,mscale=0.1):
        self.p = p
        self.mscale = mscale

    def __call__(self, x):
        y = x
        for m in range(2):
            if np.random.random() < self.p:
                s = (self.mscale*np.random.random()) + 1.0
                y[m,:,3] = s*x[m,:,3]
        return y
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, mscale={1})'.format(self.p, self.mscale)

class FreiburgRGBDDataset(Dataset):

    def __init__(self, root_dir, mode='train', color=True, transform=None, seed=None):
        self.root_dir = root_dir
        self.transform = transform
        self.color = color
        self.mode = mode

        data = np.load(os.path.join(self.root_dir,'/imatge/jdelarica/work/PycharmProjects/TFG/data/teddy/x.npy'))
        labels = np.load(os.path.join(self.root_dir,'/imatge/jdelarica/work/PycharmProjects/TFG/data/teddy/y.npy'))

        if seed:
            np.random.seed(seed)
        Ntotal = labels.shape[0]
        indices = np.random.permutation(Ntotal)
        train_len = int(Ntotal*0.8)
        val_len = int(Ntotal*0.1)
        test_len = Ntotal - train_len - val_len
        train_idx, val_idx, test_idx = indices[:train_len], indices[train_len:train_len+val_len], indices[train_len+val_len:]
        if self.mode == 'train':
            idx = train_idx
        if self.mode == 'val':
            idx = val_idx
        if self.mode == 'test':
            idx = test_idx
        self.data = data[idx,:,:,:]
        self.labels = labels[idx]
        if seed:
            np.random.seed(None)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if self.color:
            x = self.data[idx,:,:,:]
        else:
            x = self.data[idx,:,:,0:3]
            
        if self.transform:
            x = self.transform(x)
        y = np.array(self.labels[idx])
        
        # Convert to torch FloatTensor
        xt = torch.from_numpy(x.flatten()).float()
        yt = torch.from_numpy(y.flatten()).float()
        return xt,yt

    def get_input_size(self):
        if self.color:
            x = self.data[0,:,:,:].flatten()
        else:
            x = self.data[0,:,:,0:3].flatten()
        return (len(x))
