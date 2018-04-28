# Audiovisual Systems Engineering Final Thesis 
# 3D Point Cloud Correspondences using Deep Learning
#
# Barcelona School of Telecommunications Engineering 
# Universitat Politècnica de Catalunya
#
# ply2shelf.py reads the PLY files from the database, and computes a dictionary shelf file.
# 
# (x, y, z, r, g, b) coordinates are set to (x, y, z, Y), using the Y = 0.2126*R + 0.7152*G + 0.0722*B luminance computation.
#
# Author: Javier de la Rica
# Date: November 2017

from plyfile import PlyData
import os
import shelve
import numpy as np
path = '/work/jdelarica/datasets/freiburg1_teddy'

fileList = []
fileDir = os.walk(path)

for root, dirs, files in fileDir:
    for fichero in files:
        (nombreFichero, extension) = os.path.splitext(fichero)
        if(extension == ".ply"):
            fileList.append(nombreFichero+extension)

pointclouds = dict() 

d = shelve.open("/work/jdelarica/dict_freiburg1_teddy/pointclouds.shelf")

for i in range(len(fileList)):
    pointcloud = PlyData.read('/work/jdelarica/datasets/freiburg1_teddy/' + fileList[i])
    pclength = pointcloud.elements[0].count
    key = '' + fileList[i][5:9] + ''
    Y_vector = np.empty((pclength,1))
    pointclouds[key] = np.empty((pclength,4))
    for k in range(pclength):

        for e in range(3):
            pointclouds[key][k][e] = pointcloud['vertex'][k][e]
        pointclouds[key][k][3] = 0.2126 * pointcloud['vertex'][k][3] + 0.7152 * pointcloud['vertex'][k][4] + 0.0722 * pointcloud['vertex'][k][5]
    d[key] = pointclouds[key]
    print i

d.close()
