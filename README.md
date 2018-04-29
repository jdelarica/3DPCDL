# 3D Point Clouds Correspondences Using Deep Learning
| ![Javier de la Rica](/images/profile.jpeg) | ![Logo](/images/upc_etsetb.jpg) |
| :---: | :---: |
| Javier de la Rica | delaricajavier@gmail.com |

Using a Fully Connected Artificial Neural Network in order to binarly classify correspondences between two different three-dimensional point clouds.

### ABSTRACT
The current project arises from the motivation of studying the tools of neural networks, along with the application of the knowledge acquired during the courses in the degree in a work that brings together the training of neural networks using three-dimensional point clouds.

In the work, a system has been created, which from a previous database, computes a larger dataset making use of the information obtained from each of the point clouds key points. The objective of the project is to find correspondences between points from two three-dimensional point clouds using neural networks training.

On the one hand, as mentioned, the database of three-dimensional point clouds of the same scene has been generated from different points of view with ground truth, which later is used to compare patches of all the point clouds, in order to find correspondences between them.

On the other hand, a neural network is generated to be trained with the previously computed database as input, and observe the behavior of the comparison of point clouds given the key points mentioned above.

All in all, the main objective of the project is the study of the behavior of neural networks in a specific case, for a possible application of augmented reality (AR). Therefore, the present work is a good basis for a possible alternative work, capable of generating an augmented reality application in which three-dimensional objects are inserted into two-dimensional images.


![FCN Architecture](/images/fcn.png) 
**Figure1.** Three-layer fully connected network architecture (2fcReLu+1fcSigmoid)
