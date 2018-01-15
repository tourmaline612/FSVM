function K = kernel(X, type, para)
%   X: data matrix, each row is one observation, each column is one feature
%   type: type of kernel, can be 'simple', 'poly', or 'gaussian'
%   para: parameter for computing the 'poly' kernel, for 'simple'
%       and 'gaussian' it will be ignored
%   K: kernel matrix

if strcmp(type,'simple')
    K = X * X';
end

if strcmp(type,'poly')
    K = X * X'+ 1;
    K = K.^para;
end

if strcmp(type,'gaussian')
    K = distance_matrix(X).^2;
%     K=exp(-K./(2*para.^2));
    K = exp( -para * K );
end
