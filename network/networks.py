# Audiovisual Systems Engineering Final Thesis 
# 3D Point Cloud Correspondences using Deep Learning
#
# Barcelona School of Telecommunications Engineering 
# Universitat Politècnica de Catalunya
#
# networks.py creates a fully connected network with 'len(nodes)-1' layers
# nodes is an array with the sizes of the layer nodes. First element of
# nodes is the input size (so depends on the input data) and last element is
# the output size (in the case of a binary classification == 1)
#
# Author: Javier de la Rica
# Date: February 2018

import torch.nn as nn
import torch.nn.functional as F

class NetFC(nn.Module):
    def __init__(self,nodes=[512,100,50,1], dropout=False):
        super(NetFC, self).__init__()
        self.dropout = dropout
        self.nodes = nodes
        self.fcs = nn.ModuleList()
        for i in range(len(self.nodes)-1):
            self.fcs.append(nn.Linear(nodes[i],nodes[i+1]))
        
    def forward(self, x):
        for i in range(len(self.fcs)-1):
            x = F.relu(self.fcs[i](x))
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = F.sigmoid(self.fcs[-1](x))
        return x

