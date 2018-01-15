function [ model, L ] = FSVM_train_Sw(Ytrain, trainset, option, opts, rho)
% By using the Sw (the intra-class scatter matrix), no updating Sw
% trainset: nxd
%

% parameter settings
epsilon = opts.epsilon;
threshold = opts.threshold;
maxiter = opts.maxiter;
alpha = opts.alpha;

% calculate the intra-class scatter matrix
[S, ~] = swb( trainset, Ytrain );
[ U, E ] = eig( full(S) );
[ dummy, order ] = sort( diag(E), 'descend' );
U = U( : , order );
% semi-whiten process
dummy = dummy + epsilon;
Sigma = diag((dummy*rho).^(-alpha));
M =  U * Sigma * U';    %M=L'L
L = Sigma.^(alpha)*U';

iter = 1; stop = 1; t = 0.1; beta = 0.9;
% initialize the (w,b)
model = svmtrain( Ytrain, trainset, option);
w = calc_w(model);
rho_ = rho/norm(w);
w  = bsxfun(@rdivide, w, sqrt(sum(w.^2,1)));
while stop > threshold && iter <= maxiter
    % update M with (w,b)
    A = w * w' + epsilon * eye(size(w,1));
    gradient =  - 0.5 * inv(M) * A * inv(M) + rho_ * S;
    M_tmp = M - t * gradient + epsilon * eye(size(M,1));
    [U_tmp, E_tmp] = eig(M_tmp);
    [ dummy_M, order_M ] = sort( diag(E_tmp), 'descend' );
    U_tmp = U_tmp( : , order_M );
    Sigma_tmp = diag(dummy_M);
    U_tmp = U_tmp( : , [1:numel(dummy_M)] );
    M_update = U_tmp*Sigma_tmp*U_tmp';
    M = M_update;
    
    % update (w,b) with M and input Ztrain:kxn
    L = Sigma_tmp*U';
    Ztrain = trainset*L';
    model = svmtrain( Ytrain, Ztrain, option);
    v = calc_w(model);
    w = L' * v;
    rho_ = rho/norm(w);
    w  = bsxfun(@rdivide, w, sqrt(sum(w.^2,1)));
    
    % the objective 
    obj = -sum(model.obj) + rho_ * trace(S * M_update);
    
    % the stop criteria
    if iter > 1
        stop =  obj_previous - obj;
    end
    
    t = beta * t;
    obj_previous = obj;
    iter = iter + 1;
end
end