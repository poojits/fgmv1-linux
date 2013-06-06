#ifndef FGM_H_
#define FGM_H_
#include <math.h>
#include "linear.h"


template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#define min2(a, b)      ((a) <= (b) ? (a) : (b))

class FGM
{
public:
	FGM();
	FGM(problem *&prob_, model *&model_,const parameter *svm_param_,int max_iteration_)
	{
		max_iteration	= max_iteration_;
		param 			= svm_param_; 
		prob			= prob_;
		alpha			= model_->alpha;
		B				= svm_param_->nB;
		//we allocate the memory for the sub-features in advance
		elements		= prob->l*(svm_param_->nB+1);
		svm_model       = model_;
		FGM_allocate();
	}

	double SearchSparseElement(feature_node *xi,int index)
	{
		double node_value = 0;
		while(xi->index!=-1)
		{
			if (xi->index == index+1)
			{
				node_value = xi->value;
				break;
			}
			xi++;
		}
		return node_value;
	}

	~FGM();
	void FGM_init();
	void most_violated(int iteration);//worst-case analysis
	void most_violated_w_poly(int iteration);
	int cutting_set_evolve();
	void reset_model();
	void reset_sub_problem();
	void FGM_allocate();
	void heap_sort(weight *h,double X,int K, int i, int j);
	void sort_w2b(weight *w2b, int nB);
	void sort_rs(int *w,int n_rsi);
	void record_subfeature_sparse(weight *w2b, int iteration);
	void calculate_w2_poly();
	void calculate_w2_poly_r();
	void sort_w2b_wf(weight *w2b, int K);
	void sort_w2b_w(int K);
	void set_model();
	void svm_retrain(double eps, double Cp, double Cn, 
		int solver_type, weight *w2, int feature_num, int bias=0);



	
private:
	const parameter		*param;
	problem				*prob;
    double				*w_lin;// for linear features
	int					max_iteration;
	int                 n_ITER;
	int					elements;
	feature_node		**sub_x_space;//
	Solution			solution;
	double              *alpha;
	int					B;
	model				*svm_model;
};

#endif  // LBFGSMKL_H_
