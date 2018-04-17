function Y = vl_nnradiusloss(X, center_data, c, dzdy)
% data: D*n,    samples
% X:class_num*n,    predict values
% c: n*1,   labels

data = squeeze(X);
labels = c';
batchSize = numel(labels);
lambda = 0.1;

% Initialize the center_data for the first batch
if nargin <= 3
    % compute centerloss
    loss_centerdist = 0;
    for i = 1 : batchSize
        diff = data( :, i ) - center_data( :, labels(i) );
        loss_centerdist = loss_centerdist + sum( sum(diff.^2) );
    end
    loss_centerdist = 0.5 * loss_centerdist / batchSize;
    Y = loss_centerdist;
else
    der_X = [];
    for i = 1 : batchSize
        der_x = data(:,i) - center_data(:,labels(i));
        der_X = [ der_X, der_x ];
    end
    der_X = reshape(der_X, size(X));
    if isa(dzdy,'gpuArray')
        der_X = gpuArray(der_X) ;
    end
    Y  = lambda * der_X;
end