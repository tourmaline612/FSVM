function [bestacc,bestc,bestg] = FSVM_Cg_ForClass(train_label, train, cmin, cmax, gmin, gmax, v, cstep, gstep, opts, accstep)
% Find the optimal parameter C with v-inner fold CV for FSVM
% the semi-whitening initialization is adopted for parameter fine-tuning

%% about the parameters of SVMcg 
if nargin < 11
    accstep = 4.5;
end
if nargin < 9
    cstep = 0.8;
    gstep = 0.8;
end
if nargin < 8
    v = 5;
end
if nargin < 6
    gmax = 8;
    gmin = -8;
end
if nargin < 4
    cmax = 8;
    cmin = -8;
end
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

%% X:c Y:g cg:CVaccuracy
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
eps = 1e-1;
%% record acc with different c & g,and find the bestacc with the smallest c
bestc = 1;
bestg = 0.1;
bestacc = 0;
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ), ' -q'];
        cg(i,j) = svmtrain(train_label, Ztrain, cmd);
        
        if cg(i,j) <= 55
            continue;
        end
        
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
        
        if abs( cg(i,j)-bestacc )<=eps && bestc > basenum^X(i,j) 
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
        
    end
end
% %% to draw the acc with different c & g
% figure;
% [C,h] = contour(X,Y,cg,70:accstep:100);
% clabel(C,h,'Color','r');
% xlabel('log2c','FontSize',12);
% ylabel('log2g','FontSize',12);
% firstline = 'Parameter evaluation (contour map)[GridSearchMethod]'; 
% secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
%     ' CVAccuracy=',num2str(bestacc),'%'];
% title({firstline;secondline},'Fontsize',12);
% grid on; 
% saveas(gcf,['./figs_Fsvm/Fsvm.rbf.',setname,'.split',num2str(split),'.param1.fig'],'fig');
% 
% figure;
% meshc(X,Y,cg);
% % mesh(X,Y,cg);
% % surf(X,Y,cg);
% axis([cmin,cmax,gmin,gmax,30,100]);
% xlabel('log2c','FontSize',12);
% ylabel('log2g','FontSize',12);
% zlabel('Accuracy(%)','FontSize',12);
% firstline = 'Parameter evaluation [GridSearchMethod]'; 
% secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
%     ' CVAccuracy=',num2str(bestacc),'%'];
% title({firstline;secondline},'Fontsize',12);
% saveas(gcf,['./figs_Fsvm/Fsvm.rbf.',setname,'.split',num2str(split),'.param2.fig'],'fig');
% close all;
figure;
bar3(cg);
% axis([cmin,cmax,gmin,gmax,30,100]);
xlabel('C','FontSize',12);
ylabel('\sigma','FontSize',12);
zlabel('Accuracy(%)','FontSize',12);
firstline = ''; 
secondline = ['Best C=',num2str(bestc),' g=',num2str(bestg), ...
    ' CVAccuracy=',num2str(bestacc),'%'];
title({firstline;secondline},'Fontsize',12);
% saveas(gcf,['./figs_Fsvm/Fsvm.rbf.',setname,'.split',num2str(split),'.param2.fig'],'fig');
close all;