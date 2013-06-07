fgmv1-linux
===========
This is a linux port of the [FGM Toolkit] [1] for [Feature Generating Machine] [2].

Disclaimer
-----------
All credit goes to the original authors. I have just ported the code to Linux x64.

Original README
-----------
A linear-time SVM-type feature selection algorithm is proposed 
for large-scale and extremely high dimensional datasets, a very 
small subset of non-monotonic features can be identified from 3 
Million features for suspicious URLs prediction: 

*[Mingkui Tan, Li Wang, Ivor W. Tsang. Learning Sparse SVM for Feature 
Selection on Very High Dimensional Datasets. Proceedings of the 27th International Conference on Machine Learning (ICML 2010), Haifa, Israel, June 2010.] [3]* 



Usage Options
-----------

    -s type : set type of solver (default 1)
  	0 -- L2-regularized logistic regression (not included)
		1 -- L2-loss support vector machines (dual)(not included)
		2 -- L2-loss support vector machines (primal)(not included)
		3 -- L1-loss support vector machines (dual)(not included)
		4 -- multi-class support vector machines by Crammer and Singer(not included)
		5 -- SVMFGM: Feature Generating Machine with Linear Kernel
		6 -- LRFGM: Feature Generating Machine with logistic regression
	-c cost : set the parameter C (default 1)
	-e epsilon : set tolerance of termination criterion
		-s 0 and 2 
			|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2, 
			where f is the primal function, (default 0.01)
		-s 1, 3, and 4
			Dual maximal violation <= eps; similar to libsvm (default 0.1)

	Polynomial kernel (gamma*u'*v + coef0)^2
	-g gamma : set the parameter gamma (default 1)
	-r coef0 : set the parameter coef (default 1)
	-b bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
	-wi weight: weights adjust the parameter C of different classes
	-v n: n-fold cross validation mode
	-q : quiet mode (no outputs)

	-p : whether or not use polynomial mappings (cannot deal with very high dimensional problems)
	-t : retraining: whether or not do re-training with the selected features (default 1)
	-B : number of features: B should be greater than 1 (default 2)
	-m : number of iterations: (default 7)


Examples:
-----------
train
-

  - linear SVM
  
    ./train -s 5 -B 10 -c 5 -m 8 -t 1 - p 0 your_data_route/dataset  your_model_route/model
    
  - linear logistic regression (LR)
  
    ./train -s 6 -B 10 -c 5 -m 8 -t 1 - p 0 your_data_route/dataset  your_model_route/model
    
  - polynomial SVM
  
    ./train -s 5 -B 10 -c 5 -m 8 -t 1 - p 1 your_data_route/dataset  your_model_route/model
  
  - polynomial logistic regression (LR)
  
    ./train -s 6 -B 10 -c 5 -m 8 -t 1 - p 1 your_data_route/dataset  your_model_route/model

predict
-
  - ./predict your_test_data_route/test_data your_model_route/model your_model_route/output

Acknowledgement:
-----------
We use the [liblinear] [4] as our svm solver.
We adapt the simpleMKL solver by [Yufeng Li] [5]. The authors are very grateful for their contributions!


[1]: http://c2inet.sce.ntu.edu.sg/Mingkui/data/fgm_release.rar
[2]: http://c2inet.sce.ntu.edu.sg/Mingkui/FGM.htm
[3]: http://c2inet.sce.ntu.edu.sg/ivor/publication/FGM.pdf
[4]: http://www.csie.ntu.edu.tw/~cjlin/liblinear/
[5]: http://lamda.nju.edu.cn/liyf/
