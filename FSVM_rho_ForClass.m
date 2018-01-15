function [bestacc_rho, best_rho] = FSVM_rho_ForClass(train_label, train,  test_label, test, bestc, opts, rho_min, rho_max, rho_step)
% Find the optimal parameter rho for FSVM
% the semi-whitening initialization is adopted for parameter fine-tuning

if nargin < 7
    rho_min = -8;
    rho_max = 8;
    rho_step = 0.5;
end
if nargin < 6
    disp('Please provide the parameters for FSVM !');
    return;
end

basenum = opts.basenum;
rho = rho_min : rho_step : rho_max;
acc_rho = zeros(numel(rho),1);  
bestacc_rho = 0; best_rho = 0;
for i = 1:numel(rho)
    cmd = ['-t 0 -c ',num2str( bestc ), ' -q'];
    %% Change to different variants, keep consistant
    [ model, L ] = FSVM_train_St(train_label, train, cmd, opts, basenum^rho(i));
    Ztest = test*L';
    [~, accuracy, ~] = svmpredict(test_label, Ztest, model);
    acc_rho(i) = accuracy(1);
    
    if acc_rho(i) > bestacc_rho
        bestacc_rho = acc_rho(i);
        best_rho = basenum^rho(i);
    end
    
    if abs( acc_rho(i)-bestacc_rho )<=eps && best_rho > basenum^rho(i)
        bestacc_rho = acc_rho(i);
        best_rho = basenum^rho(i);
    end
end   
end
