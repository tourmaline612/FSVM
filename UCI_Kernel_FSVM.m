function accuracy = UCI_Kernel_FSVM()
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
%----------------------
cmin = opts.cmin;   
cmax = opts.cmax;   
cstep = opts.cstep;
%----------------------
gmin = opts.cmin;   
gmax = opts.cmax;   
gstep = opts.cstep;
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
best_c = zeros(split,1); best_g = zeros(split,1); best_r = zeros(split,1);
time_train = zeros(split,1); time_test = zeros(split,1); acc = zeros(split,1);
%% evaluate each split
for j = 1 : split
    
    disp(strcat('Training the fold ', num2str(j)));
    
    %% separate to the training and test data according to the index
    trainLabel=setdiff(1:sampleNum,kfold{j});
    testLabel=kfold{j};
    
    trainset=X(:,trainLabel)';
    testset=X(:,testLabel)';
    
    Ytrain = Y(trainLabel);
    Ytest  = Y(testLabel);
    
    %% F-SVM model learning
    % v-fold inner CV 
    disp('Cross-Validation for optimal hyper-parameters: ');
    [~,bestc, bestg] = FSVM_Cg_ForClass(Ytrain, trainset, cmin, cmax, gmin, gmax, inner_cv, cstep, gstep, opts);
    best_c(j) = bestc;
    best_g(j) = bestg;
    
    % kernel PCA
    X_tmp = [trainset; testset];
    [~, eigVector, eigValue] = kernel_PCA(X_tmp,size(X_tmp,1),'gaussian', bestg);
    SelectNum = cumsum(eigValue)./sum(eigValue);
    index = find(SelectNum >= 0.90);
    ForwardNum = index(1);
    P = eigVector(:,1:ForwardNum);
    X_tmp = kernel_PCA_NewData(X_tmp, X_tmp, P, 'gaussian',  bestg);
    trainset = X_tmp(1:size(trainset,1),:);
    testset = X_tmp(size(trainset,1)+1:end,:);
        
    disp('Validation for optimal rho:');
    [~,best_rho] = FSVM_rho_ForClass(Ytrain, trainset, Ytest, testset, bestc, opts, rho_min, rho_max, rho_step);
    best_r(j) = best_rho;
    
    tic;
    [ model, L ] = FSVM_train_St(Ytrain, trainset, ['-t 0 -c ', num2str(bestc), ' -g ',num2str(bestg),' -q'], opts, best_rho);
    time_train(j)=toc;
    
    tic;
    Ztest = testset*L';
    [~, accuracy, ~] = svmpredict(Ytest, Ztest, model);
    time_test(j) = toc;
    
    acc(j) = accuracy(1);
end
acc = mean(acc); training_time = mean(time_train); testing_time = mean(time_test);
fprintf('Dataset: %s, Accuracy: %f, Training time: %f, Testing time: %f \n', setname, acc, training_time, testing_time);
end