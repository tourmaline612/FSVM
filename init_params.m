function opts = init_params()
% set default parameters
%

% fine-tune the param C
% opts.cmin = -1;  opts.cmax = 2; opts.cstep = 1;
opts.cmin = -10; 
opts.cmax = 20; 
opts.cstep = 1;

% fine-tune the param rho
% opts.rho_min = -2; opts.rho_max = 2; opts.rho_step = 0.5;
opts.rho_min = -8; 
opts.rho_max = 8; 
opts.rho_step = 0.5;

% fine-tune the param sigma
% opts.gmin = -2; opts.gmax = 1; opts.gstep = 1;
opts.gmin = -20; 
opts.gmax = 10; 
opts.gstep = 1;

% inner CV and running times
opts.inner_cv = 5; % inner CV to determin the optimal parameters
opts.split = 10;

% other params
opts.basenum = 2;       % the base number to fine-tune the parameter rho
opts.epsilon = 1e-5;
opts.threshold = 1e-3;  
opts.maxiter = 15;      % the iteration number of FSVM solver
opts.coef = 0.1;
opts.alpha = 0.5;

end