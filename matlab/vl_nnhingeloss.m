function y = vl_nnhingeloss(x,c,varargin)
% Calculate the hinge loss for multiple classification
%
% xiaohe wu, 2018.04.16

if ~isempty(varargin) && ~ischar(varargin{1})  % passed in dzdy
  dzdy = varargin{1} ;
  varargin(1) = [] ;
else
  dzdy = [] ;
end

opts.instanceWeights = [] ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

inputSize = [size(x,1) size(x,2) size(x,3) size(x,4)] ;
labels = c;

% Form 1: C has one label per image. In this case, get C in form 2 or
% form 3.
c = gather(c) ;
if numel(c) == inputSize(4)
  c = reshape(c, [1 1 1 inputSize(4)]) ;
  c = repmat(c, inputSize(1:2)) ;
end

hasIgnoreLabel = any(c(:) == 0);

% --------------------------------------------------------------------
% Spatial weighting
% --------------------------------------------------------------------

% work around a bug in MATLAB, where native cast() would slow
% progressively
if isa(x, 'gpuArray')
  switch classUnderlying(x) ;
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
else
  switch class(x)
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
end

labelSize = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
assert(isequal(labelSize(1:2), inputSize(1:2))) ;
assert(labelSize(4) == inputSize(4)) ;
instanceWeights = [] ;

assert(labelSize(3) == 1) ;

if hasIgnoreLabel
    % null labels denote instances that should be skipped
    instanceWeights = cast(c(:,:,1,:) ~= 0) ;
end

if ~isempty(opts.instanceWeights)
  % important: this code needs to broadcast opts.instanceWeights to
  % an array of the same size as c
  if isempty(instanceWeights)
    instanceWeights = bsxfun(@times, onesLike(c), opts.instanceWeights) ;
  else
    instanceWeights = bsxfun(@times, instanceWeights, opts.instanceWeights);
  end
end

% --------------------------------------------------------------------
% Do the work
% --------------------------------------------------------------------

numPixelsPerImage = prod(inputSize(1:2)) ;
numPixels = numPixelsPerImage * inputSize(4) ;
imageVolume = numPixelsPerImage * inputSize(3) ;

n = reshape(0:numPixels-1,labelSize) ;
offset = 1 + mod(n, numPixelsPerImage) + ...
    imageVolume * fix(n / numPixelsPerImage) ;
ci = offset + numPixelsPerImage * max(c - 1,0) ;

if nargin <= 2 || isempty(dzdy)
    t = max(0, 1 - x(ci)) ;
    if ~isempty(instanceWeights)
        y = instanceWeights(:)' * t(:) ;
    else
        y = sum(t(:));
    end
else
    if ~isempty(instanceWeights)
        dzdy = dzdy * instanceWeights ;
    end
    dim = size(x,3);
    nframe = size(x,4);
    input_targets = squeeze(gather(x(ci)));
    x_tmp = squeeze(gather(x));
    gradInput_data = zeros(size(x_tmp));
    g = 1./(dim);
    for i=1:nframe
        input_target = input_targets(i);
        gradInput_target = 0;
        for j = 1 : dim
            z = 1 - input_target + x_tmp(j,i);
            if labels(i) == j
                continue;
            end
            if z>0
                h = 2*g*z;
                gradInput_target = gradInput_target - h;
                gradInput_data(j,i) = h;
            end
            
        end
        gradInput_data(labels(i),i) = gradInput_target;
    end
    y = single(reshape(gradInput_data, [1,1,size(gradInput_data)]));
    if isa(x,'gpuArray')
        y = gpuArray(y);
    end
end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.zeros(size(x),classUnderlying(x)) ;
else
  y = zeros(size(x),'like',x) ;
end

% --------------------------------------------------------------------
function y = onesLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.ones(size(x),classUnderlying(x)) ;
else
  y = ones(size(x),'like',x) ;
end
