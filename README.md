# FSVM

________
## Matlab implementations of the linear and kernel FSVM algorithm.


________
## Datasets

1. The codes take the "breast" as an example. 
2. Modify the setname variable in UCI_Linear_FSVM.m and UCI_Kernel_FSVM.m to evaluate other dataset. (UCI datasets: https://archive.ics.uci.edu/ml/datasets.html)
3. The data should be first split into the training and test data randomly (you can set more split than 10). (refer to "breast" dataset)
4. Please note the pre-process of the original data. Some of the UCI dataets have been pre-processed while some are not. Please pre-process the original data if necessary, which is important.


__________
## Description and Instructions

### Note: In the released codes, for better understanding the algorithm, four variants are provided to evaluate the utility of the total scatter matrix and the intra-class scatter matrix, with and withour updating the scatter matrix during the alternative updating stage.

* FSVM_train_St.m - GBCD algorithm to solve FSVM by using the total scatter matrix without updating the scatter matrix

* FSVM_train_update_St.m - GBCD algorithm to solve FSVM by using the total scatter matrix with updating the scatter matrix

* FSVM_train_Sw.m - GBCD algorithm to solve FSVM by using the intra-class scatter matrix without updating the scatter matrix

* FSVM_train_update_Sw.m - GBCD algorithm to solve FSVM by using the intra-class scatter matrix with updating the scatter matrix.

* calc_w.m - the calculation of w from LibSVM Toolbox

* update_St.m, update_Sw.m - update the total scatter matrix (St) and the intra-class scatter matrix (Sw) with the reweighted scatter matrix benefitted from the softmax function

* kernel_PCA.m, kernel.m, kernel_PCA_NewData, kernel_NewData, distance_matrix.m - Kernel PCA.

* swb.m - the within-class scatter matrix (Sw) and between-class scatter matrix (Sb)

* init_params.m - settings of the default parameters

* FSVM_C_ForClass.m, FSVM_Cg_ForClass.m, FSVM_rho_ForClass.m - Fine tune for optimal hyper-parameters

### Quickstart

1. Put the dataset in path ./data
2. Download the LibSVM Toolbox from http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip, complie it and addpath to the matlab workspace 
3. For the linear FSVM, run UCI_Linear_FSVM.m
4. For the kernel FSVM, run UCI_Kernel_FSVM.m
5. To evaluate differet variants, please note to change the GBCD function in corresponding place (with annotation in the codes)

________
## References

[1] X. Wu, W. Zuo*, L. Lin, W. Jia and D. Zhang."F-SVM: Combination of Feature Transformation and SVM Learning via Convex Relaxation", IEEE TNNLS 2018.

[2] H. Do, A. Kalousis, and M. Hilario, "Feature weighting using margin and radius based error bound optimization in svms", in Proc. ECML PKDD, 2009.

[3] H. Do and A. Kalousis, "Convex formulations of radius-margin based support vector machines", in Proc. ICML, 2013.

[4] P. K. Shivaswamy and T. Jebara, "Maximum relative margin and datadependent regularization", JMLR, 2010.

[5] C.-C Chang, and C.-J Lin. "LIBSVM : a library for support vector machines", ACM TIST, 2011. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm


________
## Citation

If you find the code and dataset useful in your research, please consider citing:

@article{wu2018fsvm,

  title={F-SVM: Combination of Feature Transformation and SVM Learning via Convex Relaxation},

  author={Wu, Xiaohe and Zuo*, Wangmeng and Lin, Liang and jia, Wei and Zhang, David},

  journal={IEEE Transactions on Neural Networks and Learning Systems},

  year={2018}}

________
## Contents

Feedbacks and comments are welcome! Feel free to contact us via [xhwu.cpsl.hit@gmail.com] or [angela612@126.com].


________
## Liscense
Copyright (c) 2018, Xiaohe Wu
All rights reserved. 
Redistribution and use in source and binary forms, with or without modification, are 
permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright 
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright 
  notice, this list of conditions and the following disclaimer in 
  the documentation and/or other materials provided with the distribution
        
