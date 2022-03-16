# 3D Point Clouds Correspondences Using Deep Learning 

| <img src="/images/img.jpeg" width="150"> | ![Logo](/images/upc_etsetb.jpg) |
| :---: | :---: |
| [Javier de la Rica](https://github.com/jdelarica) | delaricajavier@gmail.com |
 
Using a Fully Connected Artificial Neural Network in order to binarly classify correspondences between two different three-dimensional point clouds.
  
## Motivation | Abstract
  <p>The current project arises from the motivation of studying the tools of neural networks, along with the application of the knowledge acquired during the courses in the degree in a work that brings together the training of neural networks using three-dimensional point clouds.

In the work, a system has been created, which from a previous database, computes a larger dataset making use of the information obtained from each of the point clouds key points. The objective of the project is to find correspondences between points from two three-dimensional point clouds using neural networks training.

On the one hand, as mentioned, the database of three-dimensional point clouds of the same scene has been generated from different points of view with ground truth, which later is used to compare patches of all the point clouds, in order to find correspondences between them.

On the other hand, a neural network is generated to be trained with the previously computed database as input, and observe the behavior of the comparison of point clouds given the key points mentioned above.

All in all, the main objective of the project is the study of the behavior of neural networks in a specific case, for a possible application of augmented reality (AR). Therefore, the present work is a good basis for a possible alternative work, capable of generating an augmented reality application in which three-dimensional objects are inserted into two-dimensional images.


## 3DPCDL Architecture
```
.
|— data
|    |— database.py
|    |— database_prepare.py
|    |— ply2shelf.py
|
|— network
|     |— datasets.py
|     |— networks.py
|     |— test.py
|     |— train.py
|
|— research
|     |— find-neighbors.m
|     |— plot_sphere.m
|
|— results
|     |— tests_512_100_50_1
|     |— tests_512_120_84_1_V1
|     |— tests_512_120_84_1_V2
|     	   |— sgd_lr0.0001_mom0.0_wd0.0_color 
|     		|— best.txt (best accuracy obtained for this network architecture)
|     		|— net.pkl
|     		|— train.py
|     		|— train.tct
|     		|— train_accu.png
|     		|— train_loss.png
```
### Built with
* [Python](https://www.python.org/) <br>
* [PyCharm](https://www.jetbrains.com/pycharm/)<br>
* [PyTorch](https://pytorch.org/)<br>
* [TensorFlow](https://www.tensorflow.org/)<br>
* [Image Processing Group - UPC](https://imatge.upc.edu/web/)
