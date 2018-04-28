# Audiovisual Systems Engineering Final Thesis 
# 3D Point Cloud Correspondences using Deep Learning
#
# Barcelona School of Telecommunications Engineering 
# Universitat Politècnica de Catalunya
#
# database.py reads the keypoints, matches and pointclouds dictionary files computing the database,
# saving it in a text file.
#
# Author: Javier de la Rica
# Date: November 2017

# coding=utf-8

import shelve
import numpy as np
from scipy import spatial
from random import shuffle
import pickle

# Dictionary files reading
keypoints_dict = shelve.open("/imatge/jdelarica/work/dict_freiburg1_desk/keypoints.shelf", flag='r')
matches_dict = shelve.open("/imatge/jdelarica/work/dict_freiburg1_desk/matches.shelf", flag='r')
pointclouds_dict = shelve.open("/imatge/jdelarica/work/dict_freiburg1_desk/pointclouds.shelf", flag='r')

# Set Variables
R = 0.05
N = 64
data = open('/imatge/jdelarica/work/dict_freiburg1_desk/database.txt', 'w')
n = 877720 
m = 7
database = [[0]*m for i in range(n)]
keypoints_length = len(keypoints_dict)
matches_length = len(matches_dict)

p = 0
for i in matches_dict: # i = 'pc1_pc2'
    for k in matches_dict[i]:
        k1 = k[1][0] # From ((True, (55, 32)) -> 55
        k2 = k[1][1] # From ((True, (55, 32)) -> 32

        if k[0]: # k[0] = True / False
            value = 1
        else:
            value = 0

        pc1 = i[0:4]  # From '0066_0267' -> '0066'
        pc2 = i[5:9]  # From '0066_0267' -> '0267'
        keypoint1 = keypoints_dict[pc1][k1]  # (x, y, z) from keypoints_dict['0066'][55] / Center of pc1
        keypoint2 = keypoints_dict[pc2][k2]  # (x, y, z) from keypoints_dict['0267'][32] / Center of pc2

        pc_1 = pointclouds_dict[pc1]  # ALL (x, y, z, Y) from pointclouds_dict['0066']
        pc_2 = pointclouds_dict[pc2]  # ALL (x, y, z, Y) from pointclouds_dict['0267']
        xyz_vector1 = pc_1[:, 0:3]  # ALL (x, y, z) from pointclouds_dict['0066']
        xyz_vector2 = pc_2[:, 0:3]  # ALL (x, y, z) from pointclouds_dict['0267']

        # KDTree
        point_tree1 = spatial.cKDTree(xyz_vector1)  # KDTree of ALL (x, y, z) from pointclouds_dict['0066']
        point_tree2 = spatial.cKDTree(xyz_vector2)  # KDTree of ALL (x, y, z) from pointclouds_dict['0267']
        neighbors_index1 = point_tree1.query_ball_point(keypoint1, R)
        neighbors_index2 = point_tree2.query_ball_point(keypoint2, R)

        if (len(neighbors_index1) < N or len(neighbors_index2) < N):
            print ('Not enough neighbors within the radius R for the match: ')
            print(k)
        else:
            shuffle(neighbors_index1)
            shuffle(neighbors_index2)
            indexN_neighbors1 = neighbors_index1[0:N]  # N random neighbors from the first pointcloud
            indexN_neighbors2 = neighbors_index2[0:N]  # N random neighbors from the second pointcloud

            N_neighbors1 = pc_1[indexN_neighbors1]
            N_neighbors2 = pc_2[indexN_neighbors2]

            database[p][0] = keypoint1
            database[p][1] = pc1
            database[p][2] = N_neighbors1
            database[p][3] = keypoint2
            database[p][4] = pc2
            database[p][5] = N_neighbors2
            database[p][6] = value

            print (p)
            p = p + 1
            
    pickle.dump(database, data)
    data.close()
