function [net, info] = CIFAR_FSVM(varargin)
% the entrance function to evaluate the Cifar10/100 classification
% Note: modify the datasetName in the following
%
% jessie wu, 2018.04.16
%

root = fileparts(mfilename('fullpath')) ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab', 'mex')) ;
addpath(fullfile(root, 'matlab', '+dagnn')) ;
% addpath(fullfile(root, 'matlab', 'simplenn')) ;
% addpath(fullfile(root, 'matlab', 'xtest')) ;
% addpath(fullfile(root, 'examples')) ;

if ~exist('gather')
  warning('The MATLAB Parallel Toolbox does not seem to be installed. Activating compatibility functions.') ;
  addpath(fullfile(root, 'matlab', 'compatibility', 'parallel')) ;
end

if numel(dir(fullfile(root, 'matlab', 'mex', 'vl_nnconv.mex*'))) == 0
  warning('MatConvNet is not compiled. Consider running `vl_compilenn`.');
end

% set parameters
opts.modelType = 'resnet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% Change cifar10 to cifar100 for cifar100 dataset
datasetName = 'cifar10';
switch datasetName
    case 'cifar10'
        nClasses = 10;
    case 'cifar100'
        nClasses = 100;
    otherwise
        fprintf('It should be cifar10 or cifar100 dataset! \n');
end
opts.nClasses = nClasses;
opts.expDir = fullfile(vl_rootnn, ['output_',datasetName], ...
  sprintf([datasetName,'-%s-fsvm'], opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.dataDir = fullfile([datasetName,'_data']) ;
opts.imdbPath = fullfile([datasetName,'_data'], ['imdb_',datasetName,'.mat']);

opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.meanType = 'image';
opts.border = [4 4 4 4];

opts.networkType = 'dagnn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
% set opts.train.gpus = [] for CPU!
opts.train.gpus = [1];
% if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

if numel(opts.border)~=4, 
  assert(numel(opts.border)==1);
  opts.border = ones(1,4) * opts.border;
end

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

switch opts.modelType
  case 'resnet'
    net = init_ResNet110_CIFAR_FSVM(nClasses) ;
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    if strcmp(datasetName,'cifar10')
        imdb = getCifar10Imdb(opts) ;
    elseif strcmp(datasetName,'cifar100')
        imdb = getCifar100Imdb(opts) ;
    end
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;

net.meta.dataMean = imdb.meta.dataMean; 
augData = zeros(size(imdb.images.data) + [sum(opts.border(1:2)) ...
  sum(opts.border(3:4)) 0 0], 'like', imdb.images.data); 
augData(opts.border(1)+1:end-opts.border(2), ...
  opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data; 
imdb.images.augData = augData; 
% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @train_dag_CIFAR_FSVM ;
end

[net, info] = trainfn(net, imdb, nClasses, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
if imdb.images.set(batch(1))==1,  % training
  sz0 = size(imdb.images.augData);
  sz = size(imdb.images.data);
  loc = [randi(sz0(1)-sz(1)+1) randi(sz0(2)-sz(2)+1)];
  images = imdb.images.augData(loc(1):loc(1)+sz(1)-1, ...
    loc(2):loc(2)+sz(2)-1, :, batch); 
    if rand > 0.5, images=fliplr(images) ; end
else                              % validating / testing
  images = imdb.images.data(:,:,:,batch); 
end
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = getCifar10Imdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],60000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],60000) ;
  W = z(:,set == 1)*z(:,set == 1)'/60000 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;

imdb.meta.dataMean = dataMean;
imdb.meta.meanType = opts.meanType; 
imdb.meta.whitenData = opts.whitenData;
imdb.meta.contrastNormalization = opts.contrastNormalization;

% -------------------------------------------------------------------------
function imdb = getCifar100Imdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-100-mat');
files = [{'train.mat'} {'test.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 1), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz';
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.fine_labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],60000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],60000) ;
  W = z(:,set == 1)*z(:,set == 1)'/60000 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.fine_label_names;

imdb.meta.dataMean = dataMean;
imdb.meta.meanType = opts.meanType; 
imdb.meta.whitenData = opts.whitenData;
imdb.meta.contrastNormalization = opts.contrastNormalization;


