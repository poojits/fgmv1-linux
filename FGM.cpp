#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "FGM.h"
#include "MKL.h"


#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL
#define _CRTDBG_MAP_ALLOC
//#include <crtdbg.h>

#if 1
static void info(const char *fmt,...)
{
	va_list ap;
	va_start(ap,fmt);
	vprintf(fmt,ap);
	va_end(ap);
}
static void info_flush()
{
	fflush(stdout);
}
#else
static void info(char *fmt,...) {}
static void info_flush() {}
#endif

void FGM::FGM_allocate()
{
	sub_x_space				= Malloc(feature_node *, max_iteration);
	prob->xsp				= Malloc(feature_node **, max_iteration);
	w_lin					= Malloc(double,prob->n);
	
	int i = 0;
	for (i=0;i<max_iteration;i++)
	{
		
		prob->xsp[i] = Malloc(struct feature_node *,prob->l);
	}
}

FGM::~FGM()
{
	
	for (int i = 0; i<n_ITER-1; i++)
	{
		free(sub_x_space[i]);
        free(prob->xsp[i]);
	}
	free(sub_x_space);
	free(prob->xsp);
	free(w_lin);
}

void FGM::FGM_init()
{
	int i = 0;
	int j = 0;
	for(i=0; i<prob->l; i++)
	{
		alpha[i] = 1.0;
	}

}


//calculate the final w by wighting 
void FGM::calculate_w2_poly()
{
	int i=0;
	int j=0;
	int k = 0; //for specified features
	int p = 0;
	int flag = 0;
	//initialize
	for (p = 0; p<prob->n_kernel; p++)
	{
		for (i=0;i<param->nB;i++)
		{
			svm_model->w2s[p*param->nB+i].value = svm_model->sigma[p]*svm_model->w[p*param->nB+i];
		}
	}

	p = 0;
	while (fabs(svm_model->sigma[p])<=1e-4)// record when svm_model->sigma[p]!=0
	{
		p++;
	}
	int temp_p;
	temp_p = p;
	for (i=0;i<param->nB;i++)
	{
		svm_model->solution_->w_FGM[i].index1 = svm_model->w2s[p*param->nB+i].index1;
		svm_model->solution_->w_FGM[i].index2 = svm_model->w2s[p*param->nB+i].index2;
		svm_model->solution_->w_FGM[i].value = svm_model->w2s[p*param->nB+i].value;
	}
	k = param->nB;
	svm_model->solution_->w_FGM[k].index1 = -1;// ****
	j = 0;

	for (p = p+1; p<prob->n_kernel-1; p++) // match from the second to the last
	{
		if (fabs(svm_model->sigma[p])<=1e-4)
		{
			continue;
		}
		else
		{
			for(i = 0; i<param->nB; i++)// for each elements, the new elements should be matched.
			{
				j = 0;
				flag = 0;
				while (svm_model->solution_->w_FGM[j].index1 != -1)
				{
					if ((svm_model->w2s[p*param->nB+i].index1 == svm_model->solution_->w_FGM[j].index1) 
						&& (svm_model->w2s[p*param->nB+i].index2 == svm_model->solution_->w_FGM[j].index2))
					{
						svm_model->solution_->w_FGM[j].value =  
							svm_model->solution_->w_FGM[j].value + svm_model->w2s[p*param->nB+i].value;
						flag ++;
						break;
					}
					j++;
				}
				if (flag==0)// not match
				{
					svm_model->solution_->w_FGM[j].index1 = svm_model->w2s[p*param->nB+i].index1;
					svm_model->solution_->w_FGM[j].index2 = svm_model->w2s[p*param->nB+i].index2;
					svm_model->solution_->w_FGM[j].value = svm_model->w2s[p*param->nB+i].value;
					j++;
					svm_model->solution_->w_FGM[j].index1 = -1;
				}
			}
		}
	}
	j = 0;
	while (svm_model->solution_->w_FGM[j].index1 != -1)
	{
		j++;
	}
	svm_model->feature_pair = j;
	sort_w2b(svm_model->solution_->w_FGM, svm_model->feature_pair);
}



void FGM::svm_retrain(double eps, 
							double Cp, double Cn, int solver_type, weight *w2, int feature_num, int bias)
{

	int l = prob->l;
	int n = feature_num;
	int w_size = feature_num;

	int i, s,iter = 0;//,
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double temp = 0;
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2LOSS_SVM_DUAL
	double diag_p = 0.5/Cp, diag_n = 0.5/Cn;
	double upper_bound_p = INF, upper_bound_n = INF;

	// w: 1: bias, n:linear, (n+1)n/2: quadratic
	double tmp_value;
	int j = 0;

	if(solver_type == L1LOSS_SVM_DUAL)
	{
		diag_p = 0; diag_n = 0;
		upper_bound_p = Cp; upper_bound_n = Cn;
	}

	for(i=0; i<l; i++)
	{
		{
			alpha[i] =0;
		}

		if(prob->y[i] > 0)
		{
			y[i] = +1; 
			QD[i] = diag_p;
		}
		else
		{
			y[i] = -1;
			QD[i] = diag_n;
		}
		if (bias)// add bias here
		{
			QD[i] = QD[i]+1;
		}
		j = 0;
		if (bias)
		{
			while (w2[j].index1!=-2)
			{
				tmp_value = SearchSparseElement(&prob->xsp[w2[j].indexi][i][0],w2[j].indexj);
				QD[i] += tmp_value *tmp_value;
				j++;
			}
		}else
		{
			while (w2[j].index1!=-1)
			{
				tmp_value = SearchSparseElement(&prob->xsp[w2[j].indexi][i][0],w2[j].indexj);
				QD[i] += tmp_value *tmp_value;
				j++;
			}
		}
		index[i] = i;
	}


	for(i=0; i<w_size; i++)
	{
		w2[i].value = 0;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0;s<active_size;s++)
		{
			i = index[s];
			G = 0;
			schar yi = y[i];

			j = 0;
			if (bias)
			{
				while (w2[j].index1!=-2)
				{
					tmp_value =  SearchSparseElement(&prob->xsp[w2[j].indexi][i][0],w2[j].indexj);
					G += w2[j].value*tmp_value; 
					j++;
				}
				G += w2[j].value*y[i];
			}else
			{
				while (w2[j].index1!=-1)
				{
					tmp_value =  SearchSparseElement(&prob->xsp[w2[j].indexi][i][0],w2[j].indexj);
					G += w2[j].value*tmp_value; 
					j++;
				}
			}
			G = G*yi-1;

			if(y[i] == 1)
			{
				C = upper_bound_p; 
				G += alpha[i]*diag_p; 
			}
			else 
			{
				C = upper_bound_n;
				G += alpha[i]*diag_n; 
			}

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;

				j = 0;
				if (bias)
				{
					while (w2[j].index1!=-2)
					{
						tmp_value =  SearchSparseElement(&prob->xsp[w2[j].indexi][i][0],w2[j].indexj);
						w2[j].value += d*tmp_value;//dt[xi->index-1].value*dt[xi->index-1].value;
						j++;
					}
					w2[j].value += d*1*y[i];
				}
				else
				{
					while (w2[j].index1!=-1)
					{
						tmp_value =  SearchSparseElement(&prob->xsp[w2[j].indexi][i][0],w2[j].indexj);
						w2[j].value += d*tmp_value;//dt[xi->index-1].value*dt[xi->index-1].value;
						j++;
					}
				}

			}
		}

		iter++;
		if(iter % 10 == 0)
		{
			info("."); 
			info_flush();
		}

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				printf("*"); 
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}


	// calculate objective value

	double v = 0;
	double tmp = 0;
	int nSV = 0;

	tmp = 0;
	for(i=0; i<w_size; i++)
	{	
		tmp += w2[i].value*w2[i].value;					
	}
	v += tmp;

	for(i=0; i<l; i++)
	{
		if (y[i] == 1)
			v += alpha[i]*(alpha[i]*diag_p - 2); 
		else
			v += alpha[i]*(alpha[i]*diag_n - 2);
		if(alpha[i] > 0)
			++nSV;
	}

	printf("Objective value = %lf\n",v/2);
	printf("nSV = %d\n",nSV);


	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

void FGM::set_model()
{
	if (param->t==1)
	{
		calculate_w2_poly_r();
		//Do retraining with all the selected features using SVM
		printf("do retraining with all selected features using SVM\n");
		svm_retrain(param->eps, param->C, param->C, L2LOSS_SVM_DUAL,
			svm_model->solution_->w_FGM_retrain,svm_model->feature_pair,0);
		//Do retraining on the param->nB data sets.
		printf("do retraining on the best B features using SVM\n");
		svm_retrain(param->eps, param->C, param->C, L2LOSS_SVM_DUAL,
			svm_model->solution_->w_FGM_B, param->nB+1,0);

	}else
	{
		//obtain model_->feature_pair
		calculate_w2_poly();
	}
}


int FGM::cutting_set_evolve()
{
	//FGM_init();
	most_violated(0);
	double mkl_obj = 0;
    
	MKL FGM_MKL(prob,param,1,svm_model);
	FGM_MKL.MKL_init();
    
	//train SVM with one kernel
	FGM_MKL.warm_set_model(1);
	FGM_MKL.pure_train_one();
	FGM_MKL.reset_model();
	mkl_obj = svm_model->mkl_obj;
	printf("Iteration=%d\n",0);
	printf("Objective is %f\n",mkl_obj);

	
	int iter = 1;
	double bestobj=INF;
	double ObjDif= 0.0;
    most_violated(1);
	while(iter<max_iteration-1)
	{
		printf("******************************\n");
		printf("Iteration=%d\n",iter);
		svm_model->n_kernel = iter+1;
		prob->n_kernel = iter+1;
		FGM_MKL.warm_set_model(iter+1);
		FGM_MKL.SimpleMKL();
		FGM_MKL.reset_model();
		mkl_obj = svm_model->mkl_obj;
		ObjDif = fabs(mkl_obj-bestobj);
		bestobj = mkl_obj;	
		printf("Objective is %f\n",mkl_obj);
		printf("Objective difference is %f\n",ObjDif);

		if (ObjDif<0.001*fabs(bestobj))
		{
			break;
		}
		most_violated(iter+1);

		iter++;
	}
	n_ITER = iter;
	return iter;
}





void FGM::sort_w2b_wf(weight *w2b, int K)
{
	int i;
	int j;
	int tindex1;
	int tindex2;

	int tindexi;
	int tindexj;

	double tvalue;
	for (i = 0;i<K-1;i++)//sort by i no need to sort w.value
	{
		for (j=i+1;j<K;j++)
		{
			if (w2b[i].index1>w2b[j].index1)
			{
				tindex1 = w2b[i].index1;
				tindex2 = w2b[i].index2;
				tindexi = w2b[i].indexi;
				tindexj = w2b[i].indexj;
				tvalue  = w2b[i].value;

				w2b[i].index1 = w2b[j].index1;
				w2b[i].index2 = w2b[j].index2;
				w2b[i].indexi = w2b[j].indexi;
				w2b[i].indexj = w2b[j].indexj;
				w2b[i].value  = w2b[j].value;

				w2b[j].index1 = tindex1;
				w2b[j].index2 = tindex2;
				w2b[j].indexi = tindexi;
				w2b[j].indexj = tindexj;
				w2b[j].value  = tvalue;
			}
		}
	}
	for (i = 0;i<K;i++)//sort by j 
	{
		for (j=i+1;j<K;j++)
		{
			if ((w2b[i].index1==w2b[j].index1)&&(w2b[i].index2>w2b[j].index2))
			{
				tindex1 = w2b[i].index1;
				tindex2 = w2b[i].index2;
				tindexi = w2b[i].indexi;
				tindexj = w2b[i].indexj;
				tvalue  = w2b[i].value;

				w2b[i].index1 = w2b[j].index1;
				w2b[i].index2 = w2b[j].index2;
				w2b[i].indexi = w2b[j].indexi;
				w2b[i].indexj = w2b[j].indexj;
				w2b[i].value  = w2b[j].value;

				w2b[j].index1 = tindex1;
				w2b[j].index2 = tindex2;
				w2b[j].indexi = tindexi;
				w2b[j].indexj = tindexj;
				w2b[j].value  = tvalue;
			}
		}
	}

}

//sort by i no need to sort w.value
void FGM::sort_w2b_w(int K)
{
	int i;
	int j;
	int tindex1;
	int tindex2;
	int tindexi;
	int tindexj;
	double tvalue;
	for (i = 0;i<K-1;i++)
	{
		for (j=i+1;j<K;j++)
		{
			if (fabs(svm_model->solution_->w_FGM_retrain[i].value)<
				fabs(svm_model->solution_->w_FGM_retrain[j].value))
			{
				tindex1 = svm_model->solution_->w_FGM_retrain[i].index1;
				tindex2 = svm_model->solution_->w_FGM_retrain[i].index2;
				tindexi = svm_model->solution_->w_FGM_retrain[i].indexi;
				tindexj = svm_model->solution_->w_FGM_retrain[i].indexj;
				tvalue  = svm_model->solution_->w_FGM_retrain[i].value;

				svm_model->solution_->w_FGM_retrain[i].index1 = svm_model->solution_->w_FGM_retrain[j].index1;
				svm_model->solution_->w_FGM_retrain[i].index2 = svm_model->solution_->w_FGM_retrain[j].index2;
				svm_model->solution_->w_FGM_retrain[i].indexi = svm_model->solution_->w_FGM_retrain[j].indexi;
				svm_model->solution_->w_FGM_retrain[i].indexj = svm_model->solution_->w_FGM_retrain[j].indexj;
				svm_model->solution_->w_FGM_retrain[i].value  = svm_model->solution_->w_FGM_retrain[j].value;

				svm_model->solution_->w_FGM_retrain[j].index1 = tindex1;
				svm_model->solution_->w_FGM_retrain[j].index2 = tindex2;
				svm_model->solution_->w_FGM_retrain[j].indexi = tindexi;
				svm_model->solution_->w_FGM_retrain[j].indexj = tindexj;
				svm_model->solution_->w_FGM_retrain[j].value  = tvalue;
			}
		}
	}

}


//calculate the final w by wighting 
void FGM::calculate_w2_poly_r()
{
	int i=0;
	int j=0;
	int k = 0; //for specified features
	int p = 0;
	int flag = 0;
	//initialize
	for (p = 0; p<prob->n_kernel; p++)
	{
		for (i=0;i<param->nB;i++)
		{
			svm_model->w2s[p*param->nB+i].value = svm_model->sigma[p]*svm_model->w[p*param->nB+i];
		}
	}

	p = 0;
	// record when svm_model->sigma[p]!=0
	while (fabs(svm_model->sigma[p])<=1e-4)
	{
		p++;
	}
	int temp_p;
	temp_p = p;
	for (i=0;i<param->nB;i++)
	{
		svm_model->solution_->w_FGM[i].index1 = svm_model->w2s[p*param->nB+i].index1;
		svm_model->solution_->w_FGM[i].index2 = svm_model->w2s[p*param->nB+i].index2;
		svm_model->solution_->w_FGM[i].value = svm_model->w2s[p*param->nB+i].value;
		svm_model->solution_->w_FGM_retrain[i].index1 = svm_model->w2s[p*param->nB+i].index1;
		svm_model->solution_->w_FGM_retrain[i].index2 = svm_model->w2s[p*param->nB+i].index2;
		svm_model->solution_->w_FGM_retrain[i].indexi = p;
		svm_model->solution_->w_FGM_retrain[i].indexj = i;
		svm_model->solution_->w_FGM_retrain[i].value =  svm_model->w2s[p*param->nB+i].value;
	}
	k = param->nB;
	svm_model->solution_->w_FGM[k].index1 = -1;// should notice
	svm_model->solution_->w_FGM_retrain[k].index1 = -1;
	j = 0;

	for (p = p+1; p<prob->n_kernel-1; p++) // match from the second to the last
	{
		if (fabs(svm_model->sigma[p])<=1e-4)
		{
			continue;
		}
		else
		{
			for(i = 0; i<param->nB; i++)// for each elements, the new elements should be matched.
			{
				j = 0;
				flag = 0;
				while (svm_model->solution_->w_FGM[j].index1 != -1)
				{
					if ((svm_model->w2s[p*param->nB+i].index1 == 
						svm_model->solution_->w_FGM[j].index1) && 
						(svm_model->w2s[p*param->nB+i].index2 == 
						svm_model->solution_->w_FGM[j].index2))
					{
						svm_model->solution_->w_FGM[j].value =  
							svm_model->solution_->w_FGM[j].value + svm_model->w2s[p*param->nB+i].value;
						svm_model->solution_->w_FGM_retrain[j].value = 
							svm_model->solution_->w_FGM_retrain[j].value + svm_model->w2s[p*param->nB+i].value;
						flag ++;
						break;
					}
					j++;
				}
				if (flag==0)// no match
				{
					svm_model->solution_->w_FGM[j].index1 = svm_model->w2s[p*param->nB+i].index1;
					svm_model->solution_->w_FGM[j].index2 = svm_model->w2s[p*param->nB+i].index2;
					svm_model->solution_->w_FGM[j].value = svm_model->w2s[p*param->nB+i].value;
					svm_model->solution_->w_FGM_retrain[j].indexi = p;
					svm_model->solution_->w_FGM_retrain[j].indexj = i;
					svm_model->solution_->w_FGM_retrain[j].index1 = svm_model->w2s[p*param->nB+i].index1;
					svm_model->solution_->w_FGM_retrain[j].index2 = svm_model->w2s[p*param->nB+i].index2;
					svm_model->solution_->w_FGM_retrain[j].value = svm_model->w2s[p*param->nB+i].value;
					j++;
					svm_model->solution_->w_FGM[j].index1 = -1;
					svm_model->solution_->w_FGM_retrain[j].index1 = -1;
				}
			}
		}
	}

	j = 0;
	while (svm_model->solution_->w_FGM[j].index1 != -1)
	{
		j++;
	}
	svm_model->feature_pair = j;
	// sort by w value of svm_model->solution_->w_FGM_retrain
	sort_w2b_w(svm_model->feature_pair);
	//assign the index to svm_model->solution_->w_FGM_B
	for (i = 0; i< param->nB; i++)
	{
		svm_model->solution_->w_FGM_B[i].index1 = svm_model->solution_->w_FGM_retrain[i].index1;
		svm_model->solution_->w_FGM_B[i].index2 = svm_model->solution_->w_FGM_retrain[i].index2;
		svm_model->solution_->w_FGM_B[i].indexi = svm_model->solution_->w_FGM_retrain[i].indexi;
		svm_model->solution_->w_FGM_B[i].indexj = svm_model->solution_->w_FGM_retrain[i].indexj;
		svm_model->solution_->w_FGM_B[i].value = svm_model->solution_->w_FGM_retrain[i].value;
	}
	
	svm_model->solution_->w_FGM_B[i].index1 = -1;

	// sort by index 
	sort_w2b(svm_model->solution_->w_FGM, svm_model->feature_pair);
	// sort by index: the sorting of svm_model->solution_->w_FGM is the same to svm_model->solution_->w_FGM_retrain
	sort_w2b_wf(svm_model->solution_->w_FGM_retrain, svm_model->feature_pair);
	sort_w2b_wf(svm_model->solution_->w_FGM_B, param->nB);//sort by index

}



void FGM::sort_rs(int *w,int n_rsi)
{
	int i;
	int j;
	int w_temp;
	for (i=0; i< n_rsi-1; i++)
	{
		for(j=i+1; j< n_rsi; j++)
		{
			if (w[i]>w[j])
			{
				w_temp = w[i];
				w[i] = w[j];
				w[j] = w_temp;
			}
		}
	}
}


void FGM::most_violated(int iteration)
{

	int i = 0;
	int j = 0;
	int k = 0;
	int iw = 0;
	int jw = 0;

	int Dim = prob->n;
	int active_size = prob->n;

	int n_rsi;
	int n_rsj;
	int *rsi = NULL;
	int *rsj = NULL;

	n_rsi = min(param->K,prob->n);
	n_rsj = min(param->K,prob->n);

	rsi = Malloc(int, n_rsi);
	rsj = Malloc(int, n_rsj);

	int *index = Malloc(int, prob->n);
	for(i=0;i<prob->n;i++)
	{
		w_lin[i] = 0.0;
		index[i] = i;
	}

	weight *w2b = Malloc(weight, 1*param->nB);
	for(i=0;i<1*param->nB;i++)
	{
		w2b[i].value = 0.0;
		w2b[i].index1 = 0;
		w2b[i].index2 = 0;
	}

	if (param->random == 1)
	{
		for (i=0; i<active_size; i++)
		{
			j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for (i=0; i< n_rsi; i++)
		{
			rsi[i] = index[i];
		}
		for (i=0; i<active_size; i++)
		{
			j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for (i=0; i< n_rsj; i++)
		{
			rsj[i] = index[i];
		}
		sort_rs(rsi,n_rsi);
		sort_rs(rsj,n_rsj);
	}else
	{
		n_rsi = prob->n;
		n_rsj = prob->n;
		rsi = &index[0];
		rsj = &index[0];
	}

	double sqrt2_coef0_g = sqrt(2.0)*sqrt(prob->coef0*prob->gamma);
	double sqrt2_g =  sqrt(2.0)*prob->gamma;

	int i_start = 0;
	int j_start = 0;
	double w1 = 0;//linear term
	double w11 = 0;//self quadratic term
	double w2 = 0;//quadratic term


	//step 1: Consider the linear features

	if (param->flag_poly==0)
	{
		sqrt2_coef0_g = 1.0;
	}
	for(i=0;i<prob->l;i++)
	{
		feature_node *xi = prob->x[i];
		if (alpha[i]!=0)
		{
			while (xi->index != -1&&xi->index<Dim+1)
			{
				w_lin[xi->index-1] += alpha[i]*xi->value*prob->y[i]*sqrt2_coef0_g;
				xi++;
			}
		}
	}
	for(i=0;i<n_rsi;i++)
	{
		if (fabs(w_lin[rsi[i]])>0)
		{
			heap_sort(w2b,fabs(w_lin[rsi[i]]),param->nB,rsi[i],-1);
		}
	}

	if (param->flag_poly==1)
	{
		printf("do pure polynomial kernel mappings!\n");
		long int element = (prob->n+1)*(prob->n+2)/2;
		weight *w2b_score = Malloc(weight, element);
		for (k=0;k<element;k++)
		{
			w2b_score[k].value = 0;
		}

		double w_value;
		long int hash_indx;
		weight wtemp;
		for(i=0;i<prob->l;i++)
		{
			feature_node *xj;
			feature_node *xi = prob->x[i];


			//for debugging: 
			//w_min = NULL;
			while(xi->index!= -1)
			{
				xj = xi + 1;
				//w_value = alpha[i]*xi->value*prob->y_set[i]*sqrt2_coef0_g; //i,-1
				//wtemp.index1 = xi->index-1;
				//wtemp.index2 = -1;
				//hash_indx = wtemp.index1;
				//wtemp.value = w_value;
		
				if(param->flag_poly == 1)
				{
					w_value = alpha[i]*xi->value*xi->value*prob->y[i]*prob->gamma;//i,i
					wtemp.index1 = xi->index-1;
					wtemp.index2 = xi->index-1;
					hash_indx = (wtemp.index1+1)*prob->n - wtemp.index1*(wtemp.index1+1)/2 + wtemp.index2;
					//wtemp->value = w_value;
					w2b_score[hash_indx].value = w2b_score[hash_indx].value + w_value;
					w2b_score[hash_indx].index1 = xi->index-1;
					w2b_score[hash_indx].index2 = xi->index-1;

					while(xj->index!=-1)	// quadratic (i,j)
					{
						w_value = alpha[i]*xi->value*xj->value*prob->y[i]*sqrt2_g;
						wtemp.index1 = xi->index-1;
						wtemp.index2 = xj->index-1;
						hash_indx = (wtemp.index1+1)*prob->n - wtemp.index1*(wtemp.index1+1)/2 + wtemp.index2;
						wtemp.value = w_value;
						w2b_score[hash_indx].value = w2b_score[hash_indx].value + w_value;
						w2b_score[hash_indx].index1 = xi->index-1;
						w2b_score[hash_indx].index2 = xj->index-1;

						xj++;
					}
				}
				xi++;
			}
		}
		for (k = 0; k< element; k++)
		{
			if (fabs(w2b_score[k].value)>0)
			{
				heap_sort(w2b,fabs(w2b_score[k].value),param->nB,w2b_score[k].index1,w2b_score[k].index2);
			}
		}
		free(w2b_score);
	}


	// sort by i first and then sort by j i1,i1,...,in....
	sort_w2b(w2b,param->nB);
	record_subfeature_sparse(w2b,iteration);

	free(w2b);
	if (param->random ==1)
	{
		free(rsi);
		free(rsj);
	}

	free(index);

}

void FGM::most_violated_w_poly(int iteration)
{
	int i = 0;
	int j = 0;
	int num = 0;
	int k =0;
	weight *w2b = Malloc(weight, param->nB);


	num = svm_model->feature_pair;
	int tindex1;
	int tindex2;
	double tvalue;

	// sort by svm_model->solution_->w_FGM value^2
	for (i = 0; i<param->nB; i++)
		for (j = num-1; j>=1; j--)
		{
			if (fabs(svm_model->solution_->w_FGM[j].value)>fabs(svm_model->solution_->w_FGM[j-1].value))
			{

				tindex1 = svm_model->solution_->w_FGM[j].index1;
				tindex2 = svm_model->solution_->w_FGM[j].index2;
				tvalue  = svm_model->solution_->w_FGM[j].value;
				svm_model->solution_->w_FGM[j].index1 = svm_model->solution_->w_FGM[j-1].index1;
				svm_model->solution_->w_FGM[j].index2 = svm_model->solution_->w_FGM[j-1].index2;
				svm_model->solution_->w_FGM[j].value  = svm_model->solution_->w_FGM[j-1].value;
				svm_model->solution_->w_FGM[j-1].index1 = tindex1;
				svm_model->solution_->w_FGM[j-1].index2 = tindex2;
				svm_model->solution_->w_FGM[j-1].value = tvalue;
			}
		}
		//obtain w2b
		for (i=0;i<param->nB;i++)
		{
			w2b[i].index1 = svm_model->solution_->w_FGM[i].index1;
			w2b[i].index2 = svm_model->solution_->w_FGM[i].index2;
			w2b[i].value  = svm_model->solution_->w_FGM[i].value;
		}

		// sort by i first and then sort by j i1,i1,...,in....
		sort_w2b(w2b,param->nB);


		record_subfeature_sparse(w2b,iteration);
		
		free(w2b);

}
void FGM::sort_w2b(weight *w2b, int K)
{

	int i;
	int j;
	int tindex1;
	int tindex2;
	double tvalue;
	for (i = 0;i<K-1;i++)//sort by i no need to sort w.value
	{
		for (j=i+1;j<K;j++)
		{
			if (w2b[i].index1>w2b[j].index1)
			{
				tindex1 = w2b[i].index1;
				tindex2 = w2b[i].index2;
				tvalue  = w2b[i].value;
				w2b[i].index1 = w2b[j].index1;
				w2b[i].index2 = w2b[j].index2;
				w2b[i].value  = w2b[j].value;
				w2b[j].index1 = tindex1;
				w2b[j].index2 = tindex2;
				w2b[j].value  = tvalue;
			}
		}
	}
	for (i = 0;i<K;i++)//sort by j 
	{
		for (j=i+1;j<K;j++)
		{
			if ((w2b[i].index1==w2b[j].index1)&&(w2b[i].index2>w2b[j].index2))
			{
				tindex1 = w2b[i].index1;
				tindex2 = w2b[i].index2;
				tvalue  = w2b[i].value;
				w2b[i].index1 = w2b[j].index1;
				w2b[i].index2 = w2b[j].index2;
				w2b[i].value  = w2b[j].value;
				w2b[j].index1 = tindex1;
				w2b[j].index2 = tindex2;
				w2b[j].value  = tvalue;
			}
		}
	}

}
void FGM::heap_sort(weight *h,double X,int K, int i, int j)
{
	int p = 0;
	int q = 0;
	double tv;
	int tindex1;
	int tindex2;
	if(X > h[0].value)  
	{  
		h[0].value = X;  
		h[0].index1 = i;
		h[0].index2 = j;
		int p = 0;  
		while(p < K)  
		{  
			q = 2 * p + 1;  
			if(q >= K)   
				break;  
			if((q < K - 1) && (h[q + 1].value < h[q].value))  
				q = q + 1;  
			if(h[q].value < h[p].value)  
			{  
				tv = h[p].value;
				tindex1 = h[p].index1;
				tindex2 = h[p].index2;
				h[p].index1 = h[q].index1;  
				h[p].index2 = h[q].index2;
				h[p].value = h[q].value;
				h[q].value = tv;
				h[q].index1 = tindex1;  
				h[q].index2 = tindex2;
				p = q;  
			}  
			else 
				break;  
		}  
	} 


}

int count_element(problem *&prob,const parameter *param,weight *w2s, weight *w2b, int B,int iteration)
{
	int i = 0;
	int j = 0;
	int k = 0;
	int iw = 0;
	int jw = 0;

	int i_start = 0;
	int j_start = 0;
	int iw_temp=0;
	feature_node *xi;
	feature_node *xj;

	double sqrt2_coef0_g = sqrt(2.0)*sqrt(prob->coef0*prob->gamma);
	double sqrt2_g =  sqrt(2.0)*prob->gamma;

	long int element=0;
	if (param->solver_type == SVMFGM && param->flag_poly == 0)
	{

		for(k=0;k<prob->l;k++)
		{
			i_start = 0;
			xi = prob->x[k];
			j = 0;
			for(i=0; i<B; i++)
			{
				iw = w2b[i].index1;
				if(xi[i_start].index==-1)
				{
					break;
				}
				while(xi[i_start].index!=-1&&xi[i_start].index-1<iw)
				{
					i_start++;
				}
				if(xi[i_start].index-1==iw)
				{
					element++;
					j++;
				}
			}

		}
	
		return element;
	}


	// if polynomial kernel is used
	for(k=0;k<prob->l;k++)
	{
		xi = prob->x[k];
		xj = prob->x[k];
		i_start = 0;
		j_start = 0;

		j = 0;
		for(i=0;i<B;i++)
		{
			iw = w2b[i].index1;
			jw = w2b[i].index2;

			// for the linear features where w2b[iw].index2==-1
			if (jw==-1)
			{
				//record the linear feature here
				//find the corresponding xi and xj
				while(xi[i_start].index-1<iw&&xi[i_start].index!=-1)
				{
					i_start++;
				}
				if (xi[i_start].index==-1)
				{
					break;
				}

				if (xi[i_start].index-1==iw)
				{
					element++;
					j++;
				}
			}
			else
			{

				//quadratic term
				//find the corresponding xi and xj
				while(xi[i_start].index-1<iw && xi[i_start].index!=-1)
				{
					i_start++;
				}
				if (xi[i_start].index==-1)
				{
					break;
				}
				//find the corresponding xi and xj
				while(xj[j_start].index-1<jw && xj[j_start].index!=-1)
				{
					j_start++;
				}
				if (xj[j_start].index==-1)
				{
					continue;
				}

				if (xi[i_start].index-1==iw && xj[j_start].index-1==jw)
				{
					//prob->xsp[iteration][k][j].index = i+1;
					element++;
					j++;
					j_start++;
					//iw++;
				}

			} //end for else
		}
	}

	return element;
}

void FGM::record_subfeature_sparse(weight *w2b, int iteration)
{
	int i = 0;
	int j = 0;
	int k = 0;
	int iw = 0;
	int jw = 0;

	int i_start = 0;
	int j_start = 0;
	int iw_temp=0;
	feature_node *xi;
	feature_node *xj;

	double sqrt2_coef0_g = sqrt(2.0)*sqrt(prob->coef0*prob->gamma);
	double sqrt2_g =  sqrt(2.0)*prob->gamma;

	long int element = 0;

	element = count_element(prob,param,svm_model->w2s, w2b, B,iteration);
    sub_x_space[iteration] = Malloc(feature_node , element+prob->l);
	element = 0;
	if (param->solver_type == SVMFGM && param->flag_poly == 0)
	{

		for(k=0;k<prob->l;k++)
		{
			prob->xsp[iteration][k] = &sub_x_space[iteration][element];
			i_start = 0;
			xi = prob->x[k];
			j = 0;
			for(i=0; i<B; i++)
			{
				iw = w2b[i].index1;
				if(xi[i_start].index==-1)
				{
					break;
				}
				while(xi[i_start].index!=-1&&xi[i_start].index-1<iw)
				{
					i_start++;
				}
				if(xi[i_start].index-1==iw)
				{
					prob->xsp[iteration][k][j].index = i+1;
					prob->xsp[iteration][k][j].value = xi[i_start].value;
					j++;
					element++;
				}
			}
			prob->xsp[iteration][k][j].index = -1;
			element++;

		}
		for (i = 0; i <param->nB; i++)
		{
			svm_model->w2s[iteration*param->nB+i].index1 = w2b[i].index1;
			svm_model->w2s[iteration*param->nB+i].index2 = w2b[i].index2;
			svm_model->w2s[iteration*param->nB+i].value = w2b[i].value;
		}
		return;
	}


	element = 0;

	// if polynomial kernel is used
	for(k=0;k<prob->l;k++)
	{
		xi = prob->x[k];
		xj = prob->x[k];
		i_start = 0;
		j_start = 0;

		j = 0;
		prob->xsp[iteration][k] = &sub_x_space[iteration][element];
		for(i=0;i<B;i++)
		{
			iw = w2b[i].index1;
			jw = w2b[i].index2;

			// for the linear features where w2b[iw].index2==-1
			if (jw==-1)
			{
				//record the linear feature here
				//find the corresponding xi and xj
				while(xi[i_start].index-1<iw&&xi[i_start].index!=-1)
				{
					i_start++;
				}
				if (xi[i_start].index==-1)
				{
					break;
				}

				if (xi[i_start].index-1==iw)
				{
					prob->xsp[iteration][k][j].index = i+1;
					prob->xsp[iteration][k][j].value = xi[i_start].value*sqrt2_coef0_g;
					element++;
					j++;
				}
			}
			else
			{

				//quadratic term
				//find the corresponding xi and xj
				while(xi[i_start].index-1<iw && xi[i_start].index!=-1)
				{
					i_start++;
				}
				if (xi[i_start].index==-1)
				{
					break;
				}
				//find the corresponding xi and xj
				while(xj[j_start].index-1<jw && xj[j_start].index!=-1)
				{
					j_start++;
				}
				if (xj[j_start].index==-1)
				{
					continue;
				}

				if (xi[i_start].index-1==iw && xj[j_start].index-1==jw)
				{
					prob->xsp[iteration][k][j].index = i+1;
					if (iw == jw)
					{
						prob->xsp[iteration][k][j].value = 
							xj[j_start].value *xi[i_start].value *prob->gamma;
					}
					else
					{
						prob->xsp[iteration][k][j].value = 
							xj[j_start].value *xi[i_start].value*sqrt2_g;
					}
					j++;
					element++;
					j_start++;
					//iw++;
				}

			} //end for else
		}
		prob->xsp[iteration][k][j].index = -1;
		element++;

	}

	for (i = 0; i <param->nB; i++)
	{
		svm_model->w2s[iteration*param->nB+i].index1 = w2b[i].index1;
		svm_model->w2s[iteration*param->nB+i].index2 = w2b[i].index2;
		svm_model->w2s[iteration*param->nB+i].value = w2b[i].value;
	}
}


