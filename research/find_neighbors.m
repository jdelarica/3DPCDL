% Audiovisual Systems Engineering Final Thesis 
% 3D Point Cloud Correspondences using Deep Learning
%
% Barcelona School of Telecommunications Engineering 
% Universitat Politècnica de Catalunya
%
% find_neighbors.m reads two PLY files and looks for the N neighbours within a radius
% R given two key points, corresponding to each PLY files.
%
% Author: Javier de la Rica
% Date: October 2017


PC1a = pcread('frame0000.ply');
% PC1b = pcread('frame0001.ply');

% One random point in each cloud
% [    X          Y         Z    ]
pointPC1a = [0.9783 1.698 0.843];
% pointPC1b = [0.9852 1.697 0.8524];

% Y (Luminance) from (r, g, b).
% 0 <= (r, g, b) <= 255.
a = 0.2126;
b = 0.7152;
g = 0.0722;

for j = 1:length(PC1a.Color)
    Y(j,1) = a*PC1a.Color(j,1)+b*PC1a.Color(j,2)+g*PC1a.Color(j,3);
end

% Finding all the neighbors within radius R.
R = 0.05;
N = 64;
[indicesPC1a,distsPC1a] = findNeighborsInRadius(PC1a,pointPC1a,R);

% Selecting N random values from the indices list.
indices = randperm(length(indicesPC1a));
indices = indices(1:N);
N_NeighborsPC1a = indicesPC1a(indices);

% Compute a matrix with the (x, y, z) of the correspondent indices.

for i = 1:N
    C(i,1) = PC1a.Location(N_NeighborsPC1a(i),1);
    C(i,2) = PC1a.Location(N_NeighborsPC1a(i),2);
    C(i,3) = PC1a.Location(N_NeighborsPC1a(i),3);
    C(i,4) = Y(N_NeighborsPC1a(i));
end

% Database with all the key points
keypoints = jsondecode(fileread('keypoints.json'));
