% Audiovisual Systems Engineering Final Thesis 
% 3D Point Cloud Correspondences using Deep Learning
%
% Barcelona School of Telecommunications Engineering 
% Universitat Politècnica de Catalunya
%
% plot_sphere.m reads a PLY file and looks for the N neighbours within a radius
% R given a key point as a center, plotting a 3-D sphere containing the N neighbors.
%
% Author: Javier de la Rica
% Date: April 2018

clc
clear all;

PointCloud = pcread('frame0594.ply');
pcshow(PointCloud);
KP = [-0.05761847,  0.47790131,  0.76239455];
%KP = [ 0.70962602,  0.69881076,  0.79933912]
%radius = 0.05;
%neighbors = 64;
hold on

[x y z] = sphere;

s1 = surf(0.15*x+KP(1),0.15*y+KP(2),0.15*z+KP(3), 'Marker', '.', 'EdgeColor', 'flat', 'FaceColor', 'none', 'LineStyle', ':')
hold on;
plot3(KP(1), KP(2), KP(3), '+r', 'MarkerSize', 20)