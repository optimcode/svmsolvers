#ifndef _SVMCOMMON_H_
#define _SVMCOMMON_H_

#include <cfloat>
#include <cmath>
#include <float.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <errno.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>
#include <numeric>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

using namespace std;

#define logresults(fp, fmt, ...)       \
	{                                  \
		printf(fmt, __VA_ARGS__);      \
		fprintf(fp, fmt, __VA_ARGS__); \
		fflush(stdout);                \
		fflush(fp);                    \
	}

struct alfpgm_parameter {
	int ndecomp = 0;
	int verbose = 0;
	int nWss = 0;
	double epsilon = 1e-14f;
	double mu = 0.1f;

	double theta = 0.1f;
	int npairs = 200;
	int cache_size_GB = 16;
	int min_alpha_opt = 0;
	int max_alpha_opt = 0;
	int ndecomp_threshold = 250;

	int min_alpha_opt_min_val = 5;
	int max_alpha_opt_min_val = 100;
	int min_alpha_opt_max_val = 10;
	int max_alpha_opt_max_val = 500;

	int maxfpgmiter = 500;
	int maxouteriter = 100;

	double alpha_delta_threshold = 1e-5f;
	double alpha_norm_diff_fpgm = 1e-6f;
	double accuracy = 1e-3f;
	double alfpgm_accuracy = 1e-5f;
	int max_pairs;
	int maxWSS;
	double reqAccuracy = DBL_MAX;
	double reqAccuracyFactor = 0.2;
	int max_decomp = 10000;
	int wss_type = 1;
	int total_inner_iter = 0;
	int total_outer_iter = 0;
	int max_total_iterations = 1e8;
	int fista_type = 0;
	double fista_parameters_p = 1;
	double fista_parameters_q = 1;
	double fista_parameters_r = 4;
};

struct svm_problem {
	int nData = 0;
	int numOptimized = 0;
	double *y = NULL;
	double *X = NULL;
	double *Xdot = NULL;
	bool *active = NULL;
	double *K = NULL;
	int *alpha_opt_count = NULL;
	int min_count = -1;
	double *alpha_delta = NULL;
	bool *alpha_optimized = NULL;
	vector<int> wss_index;

	vector<int> all_index;
	vector<int> wss_previous;
	vector<int> wss_current;
	double beq = 0;
	double gap;
	bool same_wss;
	bool do_not_use_same = false;
};

struct svm_parameter {
	double gamma;
	int max_nr_class = 16;
	int nr_class = 0;
	double C = 100.0f;
	double b = 0;
	int nFeatures = 0;
	int processor_count;
	const int maxpairs = 500;
	const int minpairs = 2;
	const int min_decomp_iter = 50;
	double min_check_frac = .1;
	double min_check_max = 100;
	const int cachethreshold = 20000;
};

struct svm_model {
	struct svm_parameter param; /* parameter */
	int nr_class; /* number of classes, = 2 in regression/one class svm */
	int nData; /* total #SV */
	double *X; /* X matrix */
	double **sv_coef; /* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho; /* constants in decision functions (rho[k*(k-1)/2]) */
	int *sv_indices; /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */
	int *labels; /* label of each class (label[k]) */
	int *nSV; /* number of SVs for each class (nSV[k]) */
	double *alpha;
};

#endif
