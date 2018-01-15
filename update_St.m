function S=update_St( coef, X, Z )
% reweighting benefited from the softmax function
%

Zmean=mean(Z);
Z_mean = bsxfun(@minus, Z, Zmean);

% calculate distance
dis = coef * Z_mean * Z_mean';
dis = diag(dis);

% softmax function
E = exp(bsxfun(@minus, dis, max(dis))) ;
L = sum(E) ;
dis_ratio = bsxfun(@rdivide, E, L) ;

% reweighting
X_tmp = bsxfun(@minus, X, mean(X));
X_tmp = bsxfun(@times, X_tmp, sqrt(dis_ratio));

% reweighted total scatter matrix
S = X_tmp' * X_tmp;
end