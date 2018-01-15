function [Sw, Sb]=swb( X, labels )
%X: samples
%nClass: the number of classes
%classNum: the number of samples in each class
%Sw: within-class scatter matrix£¬Sb: between-class scatter matrix

nClass = numel(unique(labels));
assert(min(labels)==1);

%calculate the Sw, Sb
Sw = 0; Sb=0;
Xmean=mean(X);
for i=1:nClass
    index = find(labels == i);
    X_perclass = X(index,:);
    X_perclass_mean = mean(X_perclass);
    X_tmp = bsxfun(@minus, X_perclass, X_perclass_mean);
    dis = sqrt(sum(X_tmp.^2,2));
    dis_radio = dis/sum(sqrt(sum(X_tmp.^2,2)));
    X_tmp = bsxfun(@times, X_tmp, sqrt(dis_radio));
    Sw = Sw + X_tmp' * X_tmp;
    Sb = Sb + (numel(index)./numel(labels)) * (X_perclass_mean-Xmean)' * (X_perclass_mean-Xmean);
end
end