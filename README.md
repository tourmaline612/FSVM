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



# CNN-FSVM

________
## Matlab implementations of the CNN-FSVM algorithm.


________
## Datasets

1. Please download the imdb.mat data. (waiting ... )
2. Put the imdb.mat data in the corresponding path /datasetName_data
3. Please download the imagenet-resnet-50-dag.mat from (http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat) and put it in the path /caltech101_data

__________
## Description and Instructions

* CIFAR_FSVM.m - The entrance function to evaluate on Cifar10 and Cifar100, please note to set the datasetName 

* init_ResNet110_CIFAR_FSVM.m - Initialize the ResNet-110 architecture with the radius-margin based loss layer

* train_dag_CIFAR_FSVM.m - Main function for network training on Cifar10 and Cifar100 datasets

* CALTECH101_FSVM.m - The entrance function to evaluate on Caltech101

* train_dag_CALTECH101_FSVM.m - Main function for network training on Caltech101 dataset

* HingeLoss.m, vl_nnhingeloss - Class and function to calculate the forward and backward of the margin based loss 

* RadiusLoss.m, vl_nnradiusloss - Class and function to calculate the forward and backward of the radius based loss

* /output_cifar10/cifar10-resnet-fsvm/trained-net-epoch-160.mat is the trained net on cifar10 with the radius-margin based loss
* /output_cifar100/cifar100-resnet-fsvm/trained-net-epoch-160.mat is the trained net on cifar100 with the radius-margin based loss
* The trained net on caltech101 can be downloaded from (waiting ...)

### Quickstart

1. The /matlab/mex files are compiled with Linux. Please download the MatCovNet Toolbox and complie it for windows, then copy the new mex files to /matlab/mex  
2. Run CNN_FSVM.m directly for cifar10 dataset
3. To evaluate differet dataset, please note to change the datasetName and learningRate in corresponding place (with annotation in the codes)

________
## References

[1] X. Wu, W. Zuo*, L. Lin, W. Jia and D. Zhang."F-SVM: Combination of Feature Transformation and SVM Learning via Convex Relaxation", IEEE TNNLS 2018.

[2] K. He, X. Zhang, S. Ren and J. Sun. "Deep Residual Learning for Image Recognition", CVPR 2016.

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
        
