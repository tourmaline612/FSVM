function net = init_ResNet110_CIFAR_FSVM(nClasses, varargin)
% initialize the ResNet-110 
%

m = 110; n = 18; opts.bottleneck = false;
networkType = 'resnet'; % 'plain' | 'resnet'
opts.reLUafterSum = true;
opts.shortcutBN = true;
opts = vl_argparse(opts, varargin); 

net = dagnn.DagNN();

% Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.weightDecay = 5e-4 ;
net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 64 ;
% different learningRate for cifar100 dataset
net.meta.trainOpts.learningRate = [0.01*ones(1,80) 0.001*ones(1,40) 0.0001*ones(1,40) 0.00001*ones(1,20) 0.000001*ones(1,20)] ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% First conv layer
block = dagnn.Conv('size',  [3 3 3 16], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = 'conv0';
net.addLayer(lName, block, 'input', lName, {[lName '_f'], [lName '_b']});

add_layer_bn(net, 16, lName, 'bn0', 0.1);
block = dagnn.ReLU('leak',0);
net.addLayer('relu0',  block, 'bn0', 'relu0');


info.lastNumChannel = 16;
info.lastIdx = 0;

% Three groups of layers
info = add_group(networkType, net, n, info, 3, 16, 1, opts);
info = add_group(networkType, net, n, info, 3, 32, 2, opts);
info = add_group(networkType, net, n, info, 3, 64, 2, opts); 

% Prediction & loss layers

if opts.reLUafterSum 
    block = dagnn.Pooling('poolSize', [8 8], 'method', 'avg', 'pad', 0, 'stride', 1);
    net.addLayer('pool_final', block, sprintf('relu%d',info.lastIdx), 'pool_final');
else
    block = dagnn.Pooling('poolSize', [8 8], 'method', 'avg', 'pad', 0, 'stride', 1);
    net.addLayer('pool_final', block, sprintf('sum%d',info.lastIdx), 'pool_final');
end

block = dagnn.Conv('size', [1 1 info.lastNumChannel info.lastNumChannel], 'hasBias', true, ...
                   'stride', 1, 'pad', 0);
lName = sprintf('transform');
net.addLayer(lName, block, 'pool_final', lName, {[lName '_f'], [lName '_b']});

block = dagnn.BatchNorm('numChannels', info.lastNumChannel);
lName = sprintf('transform_bn');
net.addLayer(lName, block, 'transform',lName, ...
  {[lName '_g'], [lName '_b'], [lName '_m']});
pidx = net.getParamIndex({[lName '_g'], [lName '_b'], [lName '_m']});
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(2)).weightDecay = 0; 
net.params(pidx(3)).learningRate = 0.1;
net.params(pidx(3)).trainMethod = 'average'; 

block = dagnn.Conv('size', [1 1 info.lastNumChannel nClasses], 'hasBias', true, ...
                   'stride', 1, 'pad', 0);
lName = sprintf('prediction');
net.addLayer(lName, block, 'transform_bn', lName, {[lName '_f'], [lName '_b']});

net.addLayer('hingeloss', ...
             dagnn.HingeLoss('loss', 'hingeloss') ,...
             {'prediction', 'label'}, ...
             'Lw_obj') ;
         
net.addLayer('radiusloss', ...
             dagnn.RadiusLoss('loss', 'radiusloss') ,...
             {'transform_bn','center_data', 'label'}, ...
             'LM_obj') ;

net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'prediction', 'label'}, ...
             'top1error') ;
         
net.initParams();



% Add a group of layers containing 2n/3n conv layers
function info = add_group(netType, net, n, info, w, ch, stride, opts)
if strcmpi(netType, 'plain'), 
  if isfield(info, 'lastName'), 
    lName = info.lastName; 
    info = rmfield(info, 'lastName');
  else
    lName = sprintf('relu%d', info.lastIdx);
  end
  add_block_conv(net, sprintf('%d', info.lastIdx+1), lName, ...
    [w w info.lastNumChannel ch], stride, opts); 
  info.lastIdx = info.lastIdx + 1;
  info.lastNumChannel = ch;
  for i=2:2*n,
    add_block_conv(net, sprintf('%d', info.lastIdx+1), sprintf('relu%d', info.lastIdx), ...
      [w w ch ch], 1, opts);
    info.lastIdx = info.lastIdx + 1;
  end
elseif strcmpi(netType, 'resnet'), 
  info = add_block_res(net, info, [w w info.lastNumChannel ch], stride, true, opts); 
  for i=2:n, 
        if opts.bottleneck,
            info = add_block_res(net, info, [w w 4*ch ch], 1, false, opts);
        else
            info = add_block_res(net, info, [w w ch ch], 1, false, opts);
        end 
  end
end


% Add a smallest residual unit (2/3 conv layers)
function info = add_block_res(net, info, f_size, stride, isFirst, opts)
if isfield(info, 'lastName'), 
  lName0 = info.lastName;
  info = rmfield(info, 'lastName'); 
elseif opts.reLUafterSum || info.lastIdx == 0
    lName0 = sprintf('relu%d',info.lastIdx);
else
    lName0 = sprintf('sum%d',info.lastIdx); 
end

lName01 = lName0;
if stride > 1 || isFirst,
    if opts.bottleneck,
        ch = 4*f_size(4);
    else
        ch = f_size(4);
    end
    block = dagnn.Conv('size',[1 1 f_size(3) ch], 'hasBias', false,'stride',stride, ...
        'pad', 0);
    lName_tmp = lName0;
    lName0 = [lName_tmp '_down2'];
    net.addLayer(lName0, block, lName_tmp, lName0, [lName0 '_f']);
    
    pidx = net.getParamIndex([lName0 '_f']);
    net.params(pidx).learningRate = 0;
    
    if opts.shortcutBN ,
        add_layer_bn(net, ch, lName0, [lName01 '_d2bn'], 0.1);
        lName0 = [lName01 '_d2bn'];
    end
end

if opts.bottleneck,
    
    add_block_conv(net, sprintf('%d',info.lastIdx+1), lName01, [1 1 f_size(3) f_size(4)], stride, opts);
    info.lastIdx = info.lastIdx + 1;
    info.lastNumChannel = f_size(4);
    add_block_conv(net, sprintf('%d',info.lastIdx+1), sprintf('relu%d',info.lastIdx), ...
        [f_size(1) f_size(2) info.lastNumChannel info.lastNumChannel], 1, opts);
    info.lastIdx = info.lastIdx + 1;
    add_block_conv(net, sprintf('%d',info.lastIdx+1), sprintf('relu%d',info.lastIdx), ...
        [1 1 info.lastNumChannel info.lastNumChannel*4], 1, opts);
    info.lastIdx = info.lastIdx + 1;
    info.lastNumChannel = info.lastNumChannel*4;
    
else
    
    add_block_conv(net, sprintf('%d',info.lastIdx+1), lName01, f_size, stride, opts);
    info.lastIdx = info.lastIdx + 1;
    info.lastNumChannel = f_size(4);
    add_block_conv(net, sprintf('%d',info.lastIdx+1), sprintf('relu%d',info.lastIdx), ...
        [f_size(1) f_size(2) info.lastNumChannel info.lastNumChannel], 1, opts);
    info.lastIdx = info.lastIdx + 1;
    
end


lName1 = sprintf('bn%d', info.lastIdx);

info.lastIdx = info.lastIdx + 1;
net.addLayer(sprintf('sum%d',info.lastIdx), dagnn.Sum(), {lName0,lName1}, ...
sprintf('sum%d',info.lastIdx));

% relu
if opts.reLUafterSum
    block = dagnn.ReLU('leak', 0); 
    net.addLayer(sprintf('relu%d', info.lastIdx), block, sprintf('sum%d', info.lastIdx), ...
    sprintf('relu%d', info.lastIdx)); 
end


% Add a conv layer (followed by optional batch normalization & relu)
function net = add_block_conv(net, out_suffix, in_name, f_size, stride, opts)

block = dagnn.Conv('size',f_size, 'hasBias',false, 'stride', stride, ...
    'pad',[ceil(f_size(1)/2-0.5) floor(f_size(1)/2-0.5) ...
    ]);
lName = ['conv' out_suffix];
net.addLayer(lName, block, in_name, lName, {[lName '_f']});


add_layer_bn(net, f_size(4), lName, strrep(lName,'conv','bn'), 0.1);
lName = strrep(lName, 'conv', 'bn');

block = dagnn.ReLU('leak',0);
net.addLayer(['relu' out_suffix], block, lName, ['relu' out_suffix]);

% Add a batch normalization layer
function net = add_layer_bn(net, n_ch, in_name, out_name, lr)
block = dagnn.BatchNorm('numChannels', n_ch);
net.addLayer(out_name, block, in_name, out_name, ...
  {[out_name '_g'], [out_name '_b'], [out_name '_m']});
pidx = net.getParamIndex({[out_name '_g'], [out_name '_b'], [out_name '_m']});
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(2)).weightDecay = 0; 
net.params(pidx(3)).learningRate = lr;
net.params(pidx(3)).trainMethod = 'average'; 
