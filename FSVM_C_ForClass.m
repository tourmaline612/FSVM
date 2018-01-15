function [bestacc,bestc] = FSVM_C_ForClass(train_label, train, cmin, cmaC, v, cstep, opts)
% Find the optimal parameter C with v-inner fold CV for FSVM
% the semi-whitening initialization is adopted for parameter fine-tuning

%% about the parameters of SVMcg 
if nargin < 6
    cstep = 0.8;
end
if nargin < 5
    v = 5;
end
if nargin < 4
    cmaC = 8;
    cmin = -8;
end
%% C:c Y:g cg:CVaccuracy
C = cmin:cstep:cmaC;
[m,n] = size(C);
cg = zeros(m,n);
eps = 1e-3;
%% record acc with different c ,and find the bestacc with the smallest c
bestc = 1;
bestacc = 0;
basenum = 2;

epsilon = opts.epsilon;
rho = 1;

%trainset: nxd
trainset = bsxfun(@minus, train, mean(train));
S = cov( trainset, 1 ); % equal to cov( trainset ) * (size(trainset,1)-1)/size(trainset,1);
[ U, E ] = eig( full(S) );
[ dummy, order ] = sort( diag(E), 'descend' );
U = U( : , order );

% S2: http://ufldl.stanford.edu/wiki/index.php/%E7%99%BD%E5%8C%96
dummy = dummy + epsilon;
Sigma = diag((dummy*rho).^(-0.5));
M =  U * Sigma * U';    %M=L'L
L = Sigma^(0.5) * U';

% if L = Sigma * U', then Ztrain = trainset*L' is the PCA whitening,
% Ztrain = trainset*M is the ZCA whitening
% we define the L = Sigma^(0.5) * U', Ztrain = trainset*L' as the semi-whitening
Ztrain = trainset * L';
    
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^C(i,j) ), ' -q'];
        cg(i,j) = svmtrain( train_label, Ztrain, cmd);
        
        if cg(i,j) <= 55
            continue;
        end
        
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^C(i,j);
        end        
        
        if abs( cg(i,j)-bestacc )<=eps && bestc > basenum^C(i,j) 
            bestacc = cg(i,j);
            bestc = basenum^C(i,j);
        end        
        
    end
end
% %% to draw the acc with diffesrent c & g
% gcf = plot(cg);
% xlabel('log2c','FontSize',10);
% ylabel('Accuracy (%)','FontSize',10);
% secondline = ['Best c=',num2str(bestc), ...
%     ' CVAccuracy=',num2str(bestacc),'%'];
% firstline = 'Parameter evaluation:'; 
% title({firstline;secondline},'Fontsize',12);
% close all;     
end
