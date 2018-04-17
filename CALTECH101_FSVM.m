function [net, info] = CALTECH101_FSVM(varargin)
% the entrance function to evaluate the Cifar10/100 classification
% Note: modify the datasetName in the following
%
% jessie wu, 2018.04.16
%

root = fileparts(mfilename('fullpath')) ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab', 'mex')) ;
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

opts.modelType = 'resnet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(vl_rootnn, 'output_caltech101', ...
  sprintf('caltech101-%s-fsvm', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('caltech101_data') ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb30.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'dagnn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
opts.train.gpus = [1];
% if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%      Initialize the ResNet-50 based on the Pre-trained model
% -------------------------------------------------------------------------

net = dagnn.DagNN.loadobj(load(fullfile(opts.dataDir, 'imagenet-resnet-50-dag.mat')));
net.removeLayer('res5c_relu');
net.removeLayer('pool5');
net.removeLayer('prob');
net.removeLayer('fc1000');

%
net.addLayer('conv_add', ...
             dagnn.Conv('size', [1 1 2048 512]), ...
             'res5c', ...
             'conv_add', ...
             {'conv_add_f', 'conv_add_b'}) ;
tmp_size = [1 1 2048 512];
idx = net.getParamIndex('conv_add_f');
net.params(idx).value = sqrt(2 / prod(tmp_size))*randn(tmp_size,'single');
net.params(idx).learningRate = 1;
net.params(idx).weightDecay = 1;
idx = net.getParamIndex('conv_add_b');
net.params(idx).value = zeros(tmp_size(4),1,'single') ;
net.params(idx).learningRate = 1;
net.params(idx).weightDecay = 1;
%-------------------------------
net.addLayer('max_pooling' , ...
             dagnn.Pooling('poolSize', [7 7], 'method', 'max'), ...
             'conv_add', ...
             'max_pooling') ;       
%-------------------------------
net.addLayer('fc' , ...
             dagnn.Conv('size', [1 1 512 4096]), ...
             'max_pooling', ...
             'fc', ...
             {'fc_f', 'fc_b'}) ;  
tmp_size = [1 1 512 4096];
idx = net.getParamIndex('fc_f');
net.params(idx).value = sqrt(2 / prod(tmp_size))*randn(tmp_size,'single');
net.params(idx).learningRate = 1;
net.params(idx).weightDecay = 1;
idx = net.getParamIndex('fc_b');
net.params(idx).value = zeros(tmp_size(4),1,'single') ;
net.params(idx).learningRate = 1;
net.params(idx).weightDecay = 1;
%-------------------------------
net.addLayer('prediction' , ...
             dagnn.Conv('size', [1 1 4096 102]), ...
             'fc', ...
             'prediction', ...
             {'prediction_f', 'prediction_b'}) ;   
tmp_size = [1 1 4096 102];
idx = net.getParamIndex('prediction_f');
net.params(idx).value = sqrt(2 / prod(tmp_size))*randn(tmp_size,'single');
net.params(idx).learningRate = 10;
net.params(idx).weightDecay = 1;
idx = net.getParamIndex('prediction_b');
net.params(idx).value = zeros(tmp_size(4),1,'single') ;
net.params(idx).learningRate = 10;
net.params(idx).weightDecay = 1;

%-------------------------------
%-------------------------------
% loss
net.addLayer('radiusloss', ...
             dagnn.RadiusLoss('loss', 'centerloss') ,...
             {'fc','center_data', 'label'}, ...
             'LM_obj') ;
net.addLayer('hingeloss', ...
            dagnn.HingeLoss('loss', 'mhinge') ,...
            {'prediction', 'label'}, ...
            'Lw_obj') ;
net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'prediction', 'label'}, ...
             'top1error') ;
net.addLayer('top5error', ...
             dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
             {'prediction', 'label'}, ...
             'top5error') ;
net.renameVar('data', 'input');

% Meta parameters
net.meta.inputSize = [224 224 3] ;
net.meta.trainOpts.weightDecay = 5e-4 ;
net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 64 ;
net.meta.trainOpts.learningRate = [0.01*ones(1,80) 0.001*ones(1,80) 0.0001*ones(1,40)] ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;


if exist(opts.imdbPath, 'file')
  load(opts.imdbPath) ;
else
  imdb = getCifar10Imdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;

opts.meanType = 'image';
opts.border = [4 4 4 4];
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
  case 'dagnn', trainfn = @train_dag_CALTECH101_FSVM ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
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
  img = imdb.images.augData(loc(1):loc(1)+sz(1)-1, ...
    loc(2):loc(2)+sz(2)-1, :, batch); 
  for i = 1:size(img,4)
    images(:,:,:,i) = imresize(img(:,:,:,i), [224, 224]);
  end
  if rand > 0.5, images=fliplr(images) ; end
else                              % validating / testing
  img = imdb.images.data(:,:,:,batch); 
  for i = 1:size(img,4)
    images(:,:,:,i) = imresize(img(:,:,:,i), [224, 224]);
  end
end
labels = imdb.images.labels(batch) ;
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


