function accuracy = UCI_Linear_FSVM()
%---------------------------------------------------------
%---------------------------------------------------------
% For dataset:
% Data matrix X: dxn
% The fixed 10-fold: kfold

close all;
addpath('./libsvm-3.22/matlab')

dataset_path = './data/';
setname = 'breast';

% set default parameters
opts = init_params();
cmin = opts.cmin;   
cmax = opts.cmax;   
cstep = opts.cstep;
%----------------------
rho_min = opts.rho_min; 
rho_max = opts.rho_max; 
rho_step = opts.rho_step; 
%----------------------
inner_cv =opts.inner_cv;
split = opts.split;

% load the dataset
load([dataset_path,setname]);

% normalization
[X, ~] = mapminmax(X,0,1);
sampleNum  = size(X,2);
best_c = zeros(split,1); best_r = zeros(split,1);
time_train = zeros(split, 1); time_test = zeros(split, 1);
acc = zeros(split, 1);
for j = 1 : split
    
    disp(strcat('Training the fold ', num2str(j)));
    
    trainLabel=setdiff(1:sampleNum,kfold{j});
    testLabel=kfold{j};
    
    trainset=X(:,trainLabel)';
    testset=X(:,testLabel)';
    
%     % PCA if necessary
%     x_mean = mean(trainset,1);
%     [COEFF,SCORE, latent] = princomp(trainset);
%     SelectNum = cumsum(latent)./sum(latent);
%     index = find(SelectNum >= 0.95);
%     ForwardNum = index(1);
%     M = COEFF(:,1:ForwardNum);
%     trainset = SCORE(:,1:ForwardNum);
%     testset = bsxfun(@minus,testset,x_mean) * M;
    
    Ytrain = Y(trainLabel);
    Ytest  = Y(testLabel);
    
    %% F-SVM model learning
    % v-fold inner CV
    disp('Cross-Validation for optimal hyper-parameters: ');
    [~,bestc] = FSVM_C_ForClass(Ytrain, trainset, cmin, cmax, inner_cv, cstep, opts);
    best_c(j) = bestc;
    
    disp('Validation for optimal rho:');
    [~,best_rho] = FSVM_rho_ForClass(Ytrain, trainset, Ytest, testset, bestc, opts, rho_min, rho_max, rho_step);
    best_r(j) = best_rho;
    
    tic;
    [ model, L ] = FSVM_train_St(Ytrain, trainset, ['-t 0 -c ', num2str(bestc), '-q'], opts, best_rho);
    time_train(j)=toc;
    
    tic;
    Ztest = testset*L';
    [~, accuracy, ~] = svmpredict(Ytest, Ztest, model);
    time_test(j) = toc;
    
    acc(j) = accuracy(1);
end
acc = mean(acc);
training_time = mean(time_train);
testing_time = mean(time_test);
fprintf('Dataset: %s, Accuracy: %f, Training time: %f, Testing time: %f \n', setname, acc, training_time, testing_time);
end