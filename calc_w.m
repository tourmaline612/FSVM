function [w, b] = calc_w(model)
% calculate the w from LibSVM Toolkit
% More details can be found https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html
%

if numel(model.Label) == 2
    % for binary-class
    w =  model.SVs' * model.sv_coef;
    b = -model.rho;
    if model.Label(1) == -1
        w = -w;
        b = -b;
    end
else
    % for multi-class
    nClass = numel(model.Label);
    num_f = nClass * (nClass-1)/2;
    nSV = model.nSV;
    cum_SVs = cumsum(nSV);
    dim = size(model.SVs,2);
    w = zeros( dim, num_f);
    count = 1;
    for i = 1 : nClass - 1
        for j = i+1 : nClass
            if i == 1
                coef = [ model.sv_coef( 1 : cum_SVs(i) ,j-1); model.sv_coef( cum_SVs(j-1)+1 : cum_SVs(j) ,i) ];
                SVs = [ model.SVs( 1 : cum_SVs(i), : ); model.SVs(cum_SVs(j-1)+1 : cum_SVs(j),:)];
                w(:,count) = SVs'*coef;
                clear coef SVs;
            else
                coef = [ model.sv_coef( cum_SVs(i-1)+1 : cum_SVs(i) ,j-1); model.sv_coef( cum_SVs(j-1)+1 : cum_SVs(j) ,i) ];
                SVs = [ model.SVs(cum_SVs(i-1)+1 : cum_SVs(i),:); model.SVs(cum_SVs(j-1)+1 : cum_SVs(j),:)];
                w(:,count) = SVs'*coef;
                clear coef SVs;
            end
            count = count + 1;
        end
    end
end
end