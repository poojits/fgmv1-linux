#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include "linear.h"
#include "FGM.h"
#include <malloc.h>
typedef signed char schar;

//template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> inline void clone(T*& dst, S* src, int n)
{   
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

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

const char *solver_type_table[]=
{
	"L2_LR", "L2LOSS_SVM_DUAL", "L2LOSS_SVM","L1LOSS_SVM_DUAL","MCSVM_CS" ,"SVMFGM", "PSVMFGM", "RPFGM","SPFGM","LRFGM",NULL
};



void destroy_model(struct model *model_)
{
	if(model_->w != NULL)
		free(model_->w);
	if(model_->label != NULL)
		free(model_->label);
	if(model_->alpha != NULL)
		free(model_->alpha);
	if(model_->sigma != NULL)
		free(model_->sigma);
	if (model_->count != NULL)
		free(model_->count);

	if (model_->solution_->w_FGM !=NULL)
		free(model_->solution_->w_FGM);
	if (model_->solution_->w_FGM_retrain !=NULL)
		free(model_->solution_->w_FGM_retrain);
	if (model_->solution_->w_FGM_B !=NULL)
		free(model_->solution_->w_FGM_B);
	if (model_->solution_ != NULL)
	{
		free(model_->solution_);
	}

	free(model_);
}



model* FGM_train(struct problem *prob, const struct parameter *param)
{
	
	long int t_start = clock();
    long int t_finish;

	//data dimension
	int n = prob->n;
    prob->bias = -1;// we consider here no bias problem;

	//set the number of iteration
	int max_iteration = param->max_iteraion;

	model *model_ = Malloc(model,1);
	if(prob->bias>=0)
		model_->nr_feature = n-1;
	else
		model_->nr_feature = n;

	//initialize model
	model_->param = *param;
	model_->bias = prob->bias;
	model_->l = prob->l;
	model_->nr_class = 2;
	model_->n_kernel = 1;

	model_->label = Malloc(int, model_->nr_class);
	model_->label[0] = 1;
	model_->label[1] = -1;
	model_->nB = param->nB;
	model_->count = NULL;
	model_->alpha = NULL;

	model_->w2s							= Malloc(weight,max_iteration*(param->nB+1));
	model_->sigma						= Malloc(double,max_iteration);
	model_->w							= Malloc(double,(max_iteration)*(param->nB+1)); 
	model_->solution_					= Malloc(Solution,1);
	model_->alpha                       = Malloc(double,model_->l+1);
	model_->solution_->w_FGM			= Malloc(weight,max_iteration*param->nB+1);
	model_->solution_->w_FGM_retrain	= Malloc(weight,max_iteration*param->nB+1);
	model_->solution_->w_FGM_B			= Malloc(weight,max_iteration*param->nB+1);

	//construct FGM object:
	FGM FGM_instance(prob,model_,param,max_iteration);

	//initialize
	FGM_instance.FGM_init();
	
	//FGM training
	FGM_instance.cutting_set_evolve();

	t_finish = clock();
	printf("elaps time is %f\n",(double(t_finish- t_start)/CLOCKS_PER_SEC));
	FGM_instance.set_model();

	
	return model_;
}


int save_model_poly(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	int j;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	if (model_->param.flag_poly==1)
	{
		int w_size = (n+2)*(n+1)/2;
	}
	else
	{
		int w_size = n;
	}
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	int nr_w;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);
	fprintf(fp, "label");
	for(i=0; i<model_->nr_class; i++)
		fprintf(fp, " %d", model_->label[i]);
	fprintf(fp, "\n");

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "nB %d\n", model_->nB);

	fprintf(fp,"flag_poly %d\n", param.flag_poly);

	fprintf(fp, "coef0 %.16g\n", param.coef0);

	fprintf(fp, "gamma %.16g\n", param.gamma);

	fprintf(fp, "t %d\n", param.t);

	fprintf(fp, "feature_pair %d\n", model_->feature_pair);

	fprintf(fp, "w %d\n", model_->feature_pair);

	j = 0;
	while (model_->solution_->w_FGM[j].index1 != -1)
	{
		fprintf(fp, "%ld ", (long)model_->solution_->w_FGM[j].index1);
		fprintf(fp, "%ld:%.16g\n", (long)model_->solution_->w_FGM[j].index2,model_->solution_->w_FGM[j].value);
		j++;
	}
	if (param.t==1)
	{
		fprintf(fp, "\n");
		j = 0;
		while (model_->solution_->w_FGM_retrain[j].index1 != -1)
		{
			fprintf(fp, "%ld ", (long)model_->solution_->w_FGM_retrain[j].index1);
			fprintf(fp, "%ld:%.16g\n", (long)model_->solution_->w_FGM_retrain[j].index2,model_->solution_->w_FGM_retrain[j].value);
			j++;
		}

		fprintf(fp, "\n");
		j = 0;

		while (model_->solution_->w_FGM_B[j].index1 != -1)
		{
			fprintf(fp, "%ld ", (long)model_->solution_->w_FGM_B[j].index1);
			fprintf(fp, "%ld:%.16g\n",(long)model_->solution_->w_FGM_B[j].index2,model_->solution_->w_FGM_B[j].value);
			j++;
		}
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}
int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	int nr_w;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);
	fprintf(fp, "label");
	for(i=0; i<model_->nr_class; i++)
		fprintf(fp, " %d", model_->label[i]);
	fprintf(fp, "\n");

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				free(model_->label);
				free(model_);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model_);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;


	model_->w=Malloc(double, n*nr_w);
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

struct model *load_model_poly(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	int t;
	int nB;
	int flag_poly;
	double coef0;
	double gamma;
	int feature_pair;

	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;
	model_->w = NULL;
	model_->count = NULL;
	model_->w2s = NULL;
	model_->alpha = NULL;
	model_->sigma = NULL;
	model_->solution_ = Malloc(Solution,1);
	model_->solution_->w_FGM_retrain = NULL;
	model_->solution_->w_FGM_B = NULL;
	param.flag_poly = 0;
	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				free(model_->label);
				free(model_);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}	
		else if(strcmp(cmd,"nB")==0)
		{
			fscanf(fp,"%d",&nB);
			model_->nB = nB;
		}
		else if(strcmp(cmd,"flag_poly")==0)
		{
			fscanf(fp,"%d",&flag_poly);
			param.flag_poly = flag_poly;
		}
		else if(strcmp(cmd,"coef0")==0)
		{
			fscanf(fp,"%lf",&coef0);
			param.coef0 = coef0;
		}
		else if(strcmp(cmd,"gamma")==0)
		{
			fscanf(fp,"%lf",&gamma);
			param.gamma = gamma;
		}
		else if(strcmp(cmd,"t")==0)
		{
			fscanf(fp,"%d",&t);
			param.t = t;
		}
		else if(strcmp(cmd,"feature_pair")==0)
		{
			fscanf(fp,"%d",&feature_pair);
			model_->feature_pair = feature_pair;
		}
		else if(strcmp(cmd,"w")==0)
		{
			fscanf(fp,"%d",&feature_pair);
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model_);
			return NULL;
		}
	}
	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	if(flag_poly ==1)
	{
		int w_size = (n+2)*(n+1)/2;
	}else
	{
		int w_size = n;
	}
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;
	model_->solution_->w_FGM=Malloc(weight, model_->feature_pair+1);

	for(i=0; i<model_->feature_pair; i++)
	{
		fscanf(fp, "%d %d:%lf", &model_->solution_->w_FGM[i].index1,
			&model_->solution_->w_FGM[i].index2,&model_->solution_->w_FGM[i].value);
		fscanf(fp, "\n");
	}
	model_->solution_->w_FGM[i].index1 = -1;
	if (param.t ==1)
	{
		model_->solution_->w_FGM_retrain = Malloc(weight, model_->feature_pair+1);
		model_->solution_->w_FGM_B = Malloc(weight, model_->nB+1);
		for(i=0; i<model_->feature_pair; i++)
		{
			fscanf(fp, "%d %d:%lf", &model_->solution_->w_FGM_retrain[i].index1,
				&model_->solution_->w_FGM_retrain[i].index2,&model_->solution_->w_FGM_retrain[i].value);
			fscanf(fp, "\n");
		}
		model_->solution_->w_FGM_retrain[i].index1 = -1;

		for(i=0; i<model_->nB; i++)
		{
			fscanf(fp, "%d %d:%lf", &model_->solution_->w_FGM_B[i].index1,
				&model_->solution_->w_FGM_B[i].index2,&model_->solution_->w_FGM_B[i].value);
			fscanf(fp, "\n");
		}
		model_->solution_->w_FGM_B[i].index1 = -1;
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	if (model_->param.flag_poly==1)
	{
	int idx2;
	const feature_node *lx2;
	double *tmp_values = Malloc(double, nr_w);
	double coef0 = model_->param.coef0;
	double gamma = model_->param.gamma;
	double sqrt2 = sqrt(2.0);
	double sqrt2_coef0_g = sqrt2*sqrt(coef0*gamma);
	double sqrt2_g = sqrt2*gamma;


	for(i=0;i<nr_w;i++)
		dec_values[i] = w[i]*coef0;
	for(; (idx=lx->index)!=-1; lx++)
		// the dimension of testing data may exceed that of training
		if(idx<=n)
		{
			for (i=0; i<nr_w; i++)
				tmp_values[i] = 0;
			lx2 = lx+1;
			int w_idx = (idx*(2*n-idx+1))/2;
			for(; (idx2=lx2->index)!=-1; lx2++)
				if(idx2<=n)
					for(i=0; i<nr_w;i++)
						tmp_values[i] += w[(w_idx+idx2)*nr_w+i]*lx2->value;
			for(i=0; i<nr_w; i++)
			{
				tmp_values[i] *= sqrt2_g;
				tmp_values[i] += w[(w_idx+idx)*nr_w+i]*(lx->value)*gamma;
				tmp_values[i] += w[idx*nr_w+i]*sqrt2_coef0_g;
				dec_values[i] += tmp_values[i]*(lx->value);
			}
		}
		free(tmp_values);
	}else
	{
		for(i=0;i<nr_w;i++)
			dec_values[i] = 0;
		for(; (idx=lx->index)!=-1; lx++)
		{
			// the dimension of testing data may exceed that of training
			if(idx<=n)
				for(i=0;i<nr_w;i++)
					dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
		}
	}

	//for(i=0;i<nr_w;i++)
	//	dec_values[i] = 0;
	//for(; (idx=lx->index)!=-1; lx++)
	//{
	//	// the dimension of testing data may exceed that of training
	//	if(idx<=n)
	//		for(i=0;i<nr_w;i++)
	//			dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	//}

	if(nr_class==2)
	{
		if (dec_values[0]>0)
		{
			return model_->label[0];
		}else if (dec_values[0]<0)
		{
			return model_->label[1];
		}else
		{
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
			//return (model_->count[0]>model_->count[1])?model_->label[0]:model_->label[1];
		}	
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}
int predict_values_poly(const struct model *model_, const struct feature_node *x, double *dec_values, int flag=0)
{
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;

	double *tmp_values = Malloc(double, nr_w);
	double coef0 = model_->param.coef0;
	double gamma = model_->param.gamma;
	double sqrt2 = sqrt(2.0);
	double sqrt2_coef0_g = sqrt2*sqrt(coef0*gamma);
	double sqrt2_g = sqrt2*gamma;

	int iw_temp=0;
	int cursor1 =0;
	int cursor2 =0;
	int iw;
	int jw;
	const feature_node *xi;
	const feature_node *xj;
	tmp_values[0] = 0;
	dec_values[0] = 0;
	int i_start = 0;
	int j_start = 0;
	weight *w_FGM;
	int num_w;
	int bias = 0;
	if (model_->param.flag_poly==0)
	{
		sqrt2_coef0_g = 1.0;
	}

	if (flag==0)
	{
		w_FGM = model_->solution_->w_FGM;
		num_w = model_->feature_pair;
	}
	else if (flag ==1)
	{
		w_FGM = model_->solution_->w_FGM_retrain;
		num_w = model_->feature_pair;
	}
	else
	{
		w_FGM = model_->solution_->w_FGM_B;
		num_w = model_->nB+1;
		bias = 0;
	}
	if (bias)
	{
		for(i=0;i<num_w;i++)
		{
			iw = w_FGM[i].index1;
			jw = w_FGM[i].index2;
			if (iw ==-2)
			{
				tmp_values[0] = tmp_values[0] + w_FGM[i].value*1;
				continue;
			}
			if (jw==-1)// this is for the linear features where w2b[iw].index2==-1
			{
				//record the linear feature here
				xi = x;
				i_start = cursor1; //record the position of the array

				while(xi[i_start].index-1<iw&&xi[i_start].index!=-1)//find the corresponding xi and xj
				{
					i_start++;
				}
				if (xi[i_start].index==-1)
				{
					continue;
				}

				if (xi[i_start].index-1==iw)
				{
					tmp_values[0] = tmp_values[0] + w_FGM[i].value*xi[i_start].value*sqrt2_coef0_g;
				}
				cursor1 = i_start;
				iw_temp = iw;
			}
			else
			{
				if(iw>iw_temp) //if i changes
				{
					cursor2 = cursor1; //record the position of the array
				}
				//quadratic term

				xi = x;
				xj = x;
				i_start = cursor1; //record the position of the array
				j_start = cursor2;

				while(xi[i_start].index-1<iw&&xi[i_start].index!=-1)//find the corresponding xi and xj
				{
					i_start++;
				}
				if (xi[i_start].index==-1)
				{
					continue;
				}
				while(xj[j_start].index-1<jw&&xj[j_start].index!=-1)//find the corresponding xi and xj
				{
					j_start++;
				}
				if (xj[j_start].index==-1)
				{
					continue;
				}

				if (xi[i_start].index-1==iw&&xj[j_start].index-1==jw)
				{
					if (iw == jw)
					{
						tmp_values[0] = tmp_values[0] + w_FGM[i].value*xj[j_start].value *xi[i_start].value *gamma;
					}
					else
					{
						tmp_values[0] = tmp_values[0] + w_FGM[i].value*xj[j_start].value *xi[i_start].value*sqrt2_g;
					}
					j_start++;
					//iw++;
				}
				cursor1 = i_start;
				cursor2 = j_start;

				iw_temp = iw;
			} //end for else
		}
	}
	else
	{
		for(i=0;i<num_w;i++)
		{
			iw = w_FGM[i].index1;
			jw = w_FGM[i].index2;

			if (jw==-1)// this is for the linear features where w2b[iw].index2==-1
			{
				//record the linear feature here
				xi = x;
				i_start = cursor1; //record the position of the array

				while(xi[i_start].index-1<iw&&xi[i_start].index!=-1)//find the corresponding xi and xj
				{
					i_start++;
				}
				if (xi[i_start].index==-1)
				{
					continue;
				}

				if (xi[i_start].index-1==iw)
				{
					tmp_values[0] = tmp_values[0] + w_FGM[i].value*xi[i_start].value*sqrt2_coef0_g;
				}
				cursor1 = i_start;
				iw_temp = iw;
			}
			else
			{
				if(iw>iw_temp) //if i changes
				{
					cursor2 = cursor1; //record the position of the array
				}
				//quadratic term

				xi = x;
				xj = x;
				i_start = cursor1; //record the position of the array
				j_start = cursor2;

				while(xi[i_start].index-1<iw&&xi[i_start].index!=-1)//find the corresponding xi and xj
				{
					i_start++;
				}
				if (xi[i_start].index==-1)
				{
					continue;
				}
				while(xj[j_start].index-1<jw&&xj[j_start].index!=-1)//find the corresponding xi and xj
				{
					j_start++;
				}
				if (xj[j_start].index==-1)
				{
					continue;
				}

				if (xi[i_start].index-1==iw&&xj[j_start].index-1==jw)
				{
					if (iw == jw)
					{
						tmp_values[0] = tmp_values[0] + w_FGM[i].value*xj[j_start].value *xi[i_start].value *gamma;
					}
					else
					{
						tmp_values[0] = tmp_values[0] + w_FGM[i].value*xj[j_start].value *xi[i_start].value*sqrt2_g;
					}
					j_start++;
					//iw++;
				}
				cursor1 = i_start;
				cursor2 = j_start;

				iw_temp = iw;
			} //end for else
		}
	}

	dec_values[0] = tmp_values[0];
	free(tmp_values);
//else
//	{
//		for(i=0;i<nr_w;i++)
//			dec_values[i] = 0;
//		for(; (idx=lx->index)!=-1; lx++)
//		{
//			// the dimension of testing data may exceed that of training
//			if(idx<=n)
//				for(i=0;i<nr_w;i++)
//					dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
//		}
//	}

//for(i=0;i<nr_w;i++)
//	dec_values[i] = 0;
//for(; (idx=lx->index)!=-1; lx++)
//{
//	// the dimension of testing data may exceed that of training
//	if(idx<=n)
//		for(i=0;i<nr_w;i++)
//			dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
//}

	if(nr_class==2)
	{
		if (dec_values[0]>0)
		{
			return model_->label[0];
		}else if (dec_values[0]<0)
		{
			return model_->label[1];
		}else
		{
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
			//return (model_->count[0]>model_->count[1])?model_->label[0]:model_->label[1];
		}	
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

int predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	int label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}
int predict_poly(const model *model_, const feature_node *x, int flag=0)
{
	double *dec_values = Malloc(double, model_->nr_class);
	int label=predict_values_poly(model_, x, dec_values, flag);
	free(dec_values);
	return label;
}
int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(model_->param.solver_type==L2_LR)
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		int label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;		
	}
	else
		return 0;
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != L2_LR
		&& param->solver_type != L2LOSS_SVM_DUAL
		&& param->solver_type != L2LOSS_SVM
		&& param->solver_type != L1LOSS_SVM_DUAL
		&& param->solver_type != MCSVM_CS
		&& param->solver_type != SVMFGM
		&& param->solver_type != PSVMFGM
		&& param->solver_type != LRFGM
		&& param->solver_type != MINLR)
		return "unknown solver type";

	return NULL;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, int *target)
{
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);

	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(int,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		//struct model *submodel = train(&subprob,param);
		//needs to be further modefied 
		struct model *submodel; 
		for(j=begin;j<end;j++)
			target[perm[j]] = predict(submodel,prob->x[perm[j]]);
		destroy_model(submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}
