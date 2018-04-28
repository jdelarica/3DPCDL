# 3D Point Clouds Correspondences Using Deep Learning
| ![Javier de la Rica](/images/Javier.JPG) | ![Logo](/images/upc_etsetb.jpg) |
| :---: | :---: |
| Javier de la Rica | delaricajavier@gmail.com |

Using a Fully Connected Artificial Neural Network in order to binarly classify correspondences between two different three-dimensional point clouds.

 
 
### Creacion base de datos 2

#### Paso 1. 
Script: pointclouds_dict.py
A partir de las imagenes .ply registered, generamos un archivo .shelf donde se almacena el dataset en forma de diccionario.
#### Paso 2.
Script: database.py
Debo hacer uso del .shelf que acabo de generar, y los matches.shelf y keyponts.shelf proporcionados.
Esto genera un database.txt, que será el archivo que utilizaré cuando entrene la red neuronal.
