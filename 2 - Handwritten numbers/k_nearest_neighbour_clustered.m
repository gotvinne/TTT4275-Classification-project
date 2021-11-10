%% Nearest neighbour clustered
clear;
rawdata = load('Dataset/data.mat');

number_of_classes = 10;
number_of_clusters = 64;

% Modify k to swap between k-nearest neighbours and nearest neighbour
% (which is just with k = 1).
k = 1;

tic;
disp('Clustering...');
data = ClusterData(rawdata, number_of_clusters, number_of_classes);

disp('Running nearest neighbour...');
[confusion_matrix, classification_table] =...
    KNearestNeighbours(data, k, number_of_classes);
toc;

%% Evaluate results
PrintResults(confusion_matrix);
PlotRandomImages(classification_table, data.test_data);