# Audiovisual Systems Engineering Final Thesis 
# 3D Point Cloud Correspondences using Deep Learning
#
# Barcelona School of Telecommunications Engineering 
# Universitat Politècnica de Catalunya
#
# database_prepare.py reads file 'database.txt' and separate into train (80%), validation (10%) and test (10%)
#
# when read, database.txt is a python list with matches between keypoints with the following info:
# 
# d[0]: [x,y,z] coordinates of the keypoint
# d[1]: 'xxx' string with keypoint info (pointcloud?)
# d[2]: 64 neighbours of the keypoint [x,y,z,c] with c grayscale color
# d[3]: [x,y,z] coordinates of second keypoint
# d[4]: 'xxx' string with keypoint info (pointcloud?)
# d[5]: 64 neighbours of the keypoint [x,y,z,c] for each of them
# d[6]: 1 if it is a match or 0 if not
#
# Author: Javier de la Rica
# Date: December 2017

from __future__ import print_function
import numpy as np
import pickle

# NOTE: This only seems to work on Python2.x :?
# database.txt is not correct, has zeros at the end so total matches is 434058 hardcoded
#
# freiburg1_desk:
Ntotal = 434058
Ngood = 119485

# freiburg1_teddy:
#Ntotal = 737114
#Ngood = 181985

N = 64 # number of neighbours
neig = [2,5] 
keyp = [0,3]
split = False

print ('Loading dataset ... ', end='')
f = open("/imatge/jdelarica/work/dict_freiburg1_teddy/database.txt", 'r')
d = pickle.load(f)
f.close()
print ('OK')

# Prepare data into 2 numpy arrays:
#
# x[i,k,n,a] and y[i] where:
#
# i: corresponds to the match [0,Ngood)
# k: keypoint [0,1]
# n: neighbour [0,N)
# a: axes x,y,z,c [0,3]
#

print ('Preparing data ... ', end='')
x = -1000*np.ones([Ngood,2,N,4], dtype=float)
y = -1000*np.ones([Ngood], dtype=float)
print ('OK')

# When copying data from 'database.txt' to array x the keypoint location is
# SUBSTRACTED from each neighbour position. Therefore neighbours in x
# do not have a global location but relative to its keypoint
print ('Copying data ({} elements) ... '.format(Ntotal), end='')
g = 0
for i in range(Ntotal):
    if int(d[i][1]) == int(d[i][4]):
        pass # this should not be here!
    else:
        for k in range(2):
            for a in range(3):
                x[g,k,:,a] = d[i][neig[k]][:,a] - d[i][keyp[k]][a]
            x[g,k,:,3] = d[i][neig[k]][:,3]
        y[g] = d[i][6]
        g += 1
print ('OK')

print(g)

# Data is normalized so mean = 0 and std = 1
print ('Normalizing data ... ', end='')
for a in range(4):
        std = np.std(x[:,:,:,a])
        mean = np.mean(x[:,:,:,a])
        x[:,:,:,a] = (x[:,:,:,a] - mean) / std
print ('OK')

if split:
        print ('Splitting into train/val/test ... ', end='')
        train_len = int(Ntotal*0.8)
        val_len = int(Ntotal*0.1)
        test_len = Ntotal - train_len - val_len
        indices = np.random.permutation(Ntotal)
        train_idx, val_idx, test_idx = indices[:train_len], indices[train_len:train_len+val_len], indices[train_len+val_len:]
        print ('OK')

        # Saving data 
        print ('Saving data ... ', end='')
        np.save('/imatge/jdelarica/work/PycharmProjects/TFG/data/teddy/x_train.npy', x[train_idx,:,:,:], allow_pickle=False)
        np.save('/imatge/jdelarica/work/PycharmProjects/TFG/data/teddy/y_train.npy', y[train_idx], allow_pickle=False)
        np.save('/imatge/jdelarica/work/PycharmProjects/TFG/data/teddy/x_val.npy',   x[val_idx,:,:,:], allow_pickle=False)
        np.save('/imatge/jdelarica/work/PycharmProjects/TFG/data/teddy/y_val.npy',   y[val_idx], allow_pickle=False)
        np.save('/imatge/jdelarica/work/PycharmProjects/TFG/data/teddy/x_test.npy',  x[test_idx,:,:,:], allow_pickle=False)
        np.save('/imatge/jdelarica/work/PycharmProjects/TFG/data/teddy/y_test.npy',  y[test_idx], allow_pickle=False)
        print('OK')

else:
        # Saving data 
        print ('Saving data ... ', end='')
        np.save('/imatge/jdelarica/work/PycharmProjects/TFG/data/teddy/x.npy', x, allow_pickle=False)
        np.save('/imatge/jdelarica/work/PycharmProjects/TFG/data/teddy/y.npy', y, allow_pickle=False)
        print('OK')
