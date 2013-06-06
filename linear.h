#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

	struct feature_node
	{
		int index;
		double value;
	};

	//To store the weight of features: used in polynomial case and other nonlinear case
	struct weight
	{
		int index1;//index 0 for single features
		int index2;//index2 = -1 indicate the linear features
		int indexi;// denote whether this component has been used, only effective for retraining
		int indexj;
		double value;// weight value
	};
	struct Solution
	{
		weight *w_FGM; 
		weight *w_FGM_retrain;
		weight *w_FGM_B; //The best B features
	};
	struct problem
	{
		int l, n, n_kernel;
		int *y;
		struct feature_node **x;
		struct feature_node ***xsp; //to store sub_features 
		double bias;            /* < 0 if no bias term */  
		int w_size;
		double coef0;
		double gamma;
		long int elements;
	};

	//5 7 9
	enum {L2_LR, L2LOSS_SVM_DUAL, L2LOSS_SVM, L1LOSS_SVM_DUAL, MCSVM_CS, SVMFGM, LRFGM, PSVMFGM, MINLR}; /* solver_type */

	struct parameter
	{
		int solver_type;
		/* these are for training only */
		double eps;	        /* stopping criteria */
		double C;
		int nr_weight;
		int *weight_label;
		double* weight;
		int initial_type;//0 for average initialization; 1 for training initialization;
		int flag_poly; //denote whether do poly or not
		int random;
		double coef0;
		double gamma;
		int max_iteraion;
		int t;
		int K; //sampling size
		int nB; //number of features;
		int fCRS;
	};

	struct model
	{
		struct parameter param;
		int nr_class;		/* number of classes */
		int nr_feature;
		double *w;
		int *label;		/* label of each class (label[n]) */
		double bias;
		double *alpha;  /* liyf 08.10.28*/
		int l;                   /* number of instance liyf 08.10.28*/

		double *sigma;
		int n_kernel;
		weight *w2s; 
		Solution *solution_;
		int *count; //for unbalanced problem
		int w_size;
		int feature_pair;
		int nB;
		double mkl_obj;
	};

	typedef struct node
	{
		weight w;
		struct node *lLink;
		struct node *rLink;
	}Wt;


	struct model* train(const struct problem *prob, const struct parameter *param);
	struct model* FGM_train(struct problem *prob, const struct parameter *param);
	struct model* LSSVM_train(struct problem *prob, const struct parameter *param);
	void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, int *target);

	int predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
	int predict_values_poly(const struct model *model_, const struct feature_node *x, double* dec_values);
	int predict(const struct model *model_, const struct feature_node *x);
	int predict_poly(const struct model *model_, const struct feature_node *x, int flag);
	int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

	int save_model(const char *model_file_name, const struct model *model_);
	int save_model_poly(const char *model_file_name, const struct model *model_);
	struct model *load_model(const char *model_file_name);
	struct model *load_model_poly(const char *model_file_name);

	int get_nr_feature(const struct model *model_);
	int get_nr_class(const struct model *model_);
	void get_labels(const struct model *model_, int* label);

	void destroy_model(struct model *model_);
	void destroy_param(struct parameter *param);
	void destroy_predict_model(struct model *model_);

	const char *check_parameter(const struct problem *prob, const struct parameter *param);

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

