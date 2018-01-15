function Sw=update_Sw( coef, X, Z, labels)
% reweighting benefited from the softmax function
%

nClass = numel(unique(labels));
assert(min(labels)==1);

Sw = 0;
for i=1:nClass
    index = find(labels == i);
    % for class i, calculate the distance to the i'th center
    Z_perclass = Z(index,:);
    Z_ = bsxfun(@minus, Z_perclass, mean(Z_perclass));
    dis = coef * Z_ * Z_';
    dis = diag(dis);
    
    % softmax function
    E = exp(bsxfun(@minus, dis, max(dis))) ;
    L = sum(E) ;
    dis_ratio = bsxfun(@rdivide, E, L) ;
    
    % reweighting
    X_perclass = X(index,:);
    X_tmp = bsxfun(@minus, X_perclass, mean(X_perclass));
    X_tmp = bsxfun(@times, X_tmp, sqrt(dis_ratio));
    
    % reweighted scatter matrix
    Sw = Sw + X_tmp' * X_tmp;
end
end