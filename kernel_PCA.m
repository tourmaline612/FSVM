function [Y, eigVector, eigValue]=kernel_PCA(X,d,type,para)
%   X: data matrix, each row is one observation, each column is one feature
%   d: reduced dimension
%   type: type of kernel, can be 'simple', 'poly', or 'gaussian'
%   para: parameter for computing the 'poly' and 'gaussian' kernel, 
%       for 'simple' it will be ignored
%   Y: dimensionanlity-reduced data
%   eigVector: eigen-vector, will later be used for pre-image
%       reconstruction

%% check input
if ( strcmp(type,'simple') || strcmp(type,'poly') || ...
        strcmp(type,'gaussian') ) == 0
    Y = [];
    eigVector = [];
    fprintf(['\nError: Kernel type ' type ' is not supported. \n']);
    return;
end

N = size(X,1);

%% kernel PCA
K0 = kernel(X, type, para);
All_One=ones(N,N)/N;
K = (K0-All_One * K0 - K0 * All_One + All_One * K0 * All_One)/N;

%% eigenvalue analysis
[V, D] = eig(K);
eigValue = diag(D);
[~,IX] = sort(eigValue,'descend');
eigVector = V(:,IX);
eigValue = eigValue(IX);

%% normailization
norm_eigVector = sqrt(sum(eigVector.^2));
eigVector = eigVector./repmat(norm_eigVector,size(eigVector,1),1);

%% dimensionality reduction
eigVector = eigVector(:,1:d);
Y = K0 * eigVector;

