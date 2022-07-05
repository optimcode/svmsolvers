#include "svmfpgm.h"

using namespace std;

int main(int argc, char **argv) {
	try {
		if (cmdOptionExists(argv, argv + argc, "-h")) {
			printf(
					"Usage: svmfpgm -f training_set_filename"
							"options:\n"
							"-e test file"
							"-a 0/1 to indicate if train fail contains gamma in third row"
							"-c cost : set the parameter C of C-SVC \n"
							"-m cachesize : set cache memory size in MB (default 100)\n"
							"-o min_check_max : set min_check_max size"
							"-u mu : mu value\n"
							"-p npairs : number of pairs"
							"-t accuracy : tolerance value\n"
							"-n alfpgm_accuracy : nral_accuracy accuracy\n"
							"-v verbose : verbose value\n"
							"-d theta : theta value\n"
							"-g gamma : gamma value\n"
							"-y processors : processors value\n"
							"-x fista_type : fista_type type value\n"
							"-z use_cache_type : use_cache_type (0,1)\n"
							"-w wss type : working set selection type (0 or 1)\n");
		}

		char *fileNameTrain;
		char *fileNameTest;
		bool file_gamma = true;
		if (cmdOptionExists(argv, argv + argc, "-f")) {
			fileNameTrain = getCmdOption(argv, argv + argc, "-f");
		} else {
			printf("File name not specified\n");
			return (-1);
		}

		if (cmdOptionExists(argv, argv + argc, "-y")) {
			svm_param.processor_count = atoi(
					getCmdOption(argv, argv + argc, "-y"));
		} else {
			svm_param.processor_count = omp_get_max_threads();

			if (svm_param.processor_count <= 0) {
				svm_param.processor_count = 2;
			} else {
				svm_param.processor_count = int(
						svm_param.processor_count / 2.0);
			}
		}

#ifdef USE_MKL
		mkl_set_num_threads(svm_param.processor_count);
		printf("Number of Processors MKL: %d\n", svm_param.processor_count);
#endif
		omp_set_num_threads(svm_param.processor_count);
		printf("Number of Processors OMP: %d\n", svm_param.processor_count);

		if (cmdOptionExists(argv, argv + argc, "-a")) {
			file_gamma = (atoi(getCmdOption(argv, argv + argc, "-a")) == 1);
		} else {
			printf("Specify file format\n");
			return (-1);
		}

		dataset train_data;
		// parameters
		alfpgm_param.verbose = 0;

		// read data and dimensions of data
		readin_Data(fileNameTrain, &train_data, file_gamma, true);
		parse_arguments(argc, argv);

		int *classes_label = NULL;
		int *start = NULL;
		int *class_count = NULL;
		int *inputdata_index = (int*) calloc(train_data.nData, sizeof(int));

		svm_problem *sub_prob = (svm_problem*) (calloc(1, sizeof(svm_problem)));

		svm_param.nr_class = svm_group_classes(&train_data, &classes_label,
				&start, &class_count, inputdata_index);

		if (svm_param.nr_class == 1) {
			printf(
					"WARNING: training data in only one class. See README for details.\n");
			return (-1);
		}

		// train k*(k-1)/2 mode alfpgm_prob->nDatas
		bool *nonzero = (bool*) (malloc(train_data.nData * sizeof(bool)));
		for (int i = 0; i < train_data.nData; i++) {
			nonzero[i] = false;
		}

		int multi_k = svm_param.nr_class * (svm_param.nr_class - 1) / 2;

		logresults(LogfileID, "Training %d classifier(s) \n", multi_k);
		logresults(LogfileID, "Fista Type %d \n", alfpgm_param.fista_type);
		logresults(LogfileID, "WSS Type %d \n", alfpgm_param.wss_type);
		decision_function *decisions_k = (decision_function*) (malloc(
				multi_k * sizeof(decision_function)));

		int classifier_index = 0;
		int i, j, k;
		int si, sj;
		int ci, cj;

		for (i = 0; i < svm_param.nr_class; i++) {
			for (j = i + 1; j < svm_param.nr_class; j++) {
				si = start[i];
				sj = start[j];
				ci = class_count[i];
				cj = class_count[j];

				sub_prob->nData = ci + cj;
				sub_prob->y = (double*) realloc(sub_prob->y,
						sizeof(double) * sub_prob->nData);
				sub_prob->X = (double*) realloc(sub_prob->X,
						sizeof(double) * sub_prob->nData * svm_param.nFeatures);
				int npositive = 0;
				for (k = 0; k < ci; k++) {
					memcpy(&sub_prob->X[k * svm_param.nFeatures],
							&train_data.X[inputdata_index[k + si]
									* svm_param.nFeatures],
							sizeof(double) * svm_param.nFeatures);
					sub_prob->y[k] = 1.0;
					npositive++;
				}

				for (k = 0; k < cj; k++) {
					memcpy(&sub_prob->X[(k + ci) * svm_param.nFeatures],
							&train_data.X[inputdata_index[k + sj]
									* svm_param.nFeatures],
							sizeof(double) * svm_param.nFeatures);
					sub_prob->y[k + ci] = -1.0;
				}

				logresults(LogfileID,
						"\nSubproblem size %d\n#positive labels : %d\n#negative labels : %d\n",
						sub_prob->nData, npositive, sub_prob->nData - npositive);
				decisions_k[classifier_index] = SVMSOLVER(sub_prob);

				if (alfpgm_param.verbose > 1) {
					logresults(LogfileID, " SVMSOLVER exit_flag %d\n",
							decisions_k[classifier_index].exit_flag);
				}

				for (k = 0; k < ci; k++) {
					if (!nonzero[si + k]
							&& fabs(decisions_k[classifier_index].alpha[k])
									> alfpgm_param.epsilon)
						nonzero[si + k] = true;
				}

				for (k = 0; k < cj; k++) {
					if (!nonzero[sj + k]
							&& fabs(decisions_k[classifier_index].alpha[ci + k])
									> alfpgm_param.epsilon)
						nonzero[sj + k] = true;
				}
				classifier_index++;
				logresults(LogfileID, "\nClassifier %d done\n",
						classifier_index);
			}
		}

		// build output
		svm_model model;

		model.nr_class = svm_param.nr_class;
		model.labels = (int*) (malloc(svm_param.nr_class * sizeof(int)));
		for (k = 0; k < svm_param.nr_class; k++) {
			model.labels[k] = classes_label[k];
		}

		model.rho = (double*) (malloc(multi_k * sizeof(double)));
		for (k = 0; k < multi_k; k++) {
			model.rho[k] = decisions_k[k].rho;
		}

		int total_sv = 0;
		int nSV;
		int *nz_count = (int*) (malloc(svm_param.nr_class * sizeof(int)));
		model.nSV = (int*) (malloc(svm_param.nr_class * sizeof(int)));
		for (int i = 0; i < svm_param.nr_class; i++) {
			nSV = 0;
			for (int j = 0; j < class_count[i]; j++) {
				if (nonzero[start[i] + j]) {
					nSV++;
					total_sv++;
				}
			}
			model.nSV[i] = nSV;
			nz_count[i] = nSV;
		}

		logresults(LogfileID, "Total nSV = %d\n", total_sv);

		model.nData = total_sv;
		model.X = (double*) (malloc(
				total_sv * svm_param.nFeatures * sizeof(double)));
		int p = 0;
		for (int i = 0; i < train_data.nData; i++) {
			if (nonzero[i]) {
				memcpy(&model.X[p * svm_param.nFeatures],
						&train_data.X[inputdata_index[i] * svm_param.nFeatures],
						sizeof(double) * svm_param.nFeatures);
				p++;
			}
		}

		int *nz_start = (int*) (malloc(svm_param.nr_class * sizeof(int)));
		nz_start[0] = 0;
		for (int i = 1; i < svm_param.nr_class; i++) {
			nz_start[i] = nz_start[i - 1] + nz_count[i - 1];
		}

		model.sv_coef = (double**) (malloc(
				(svm_param.nr_class - 1) * sizeof(double*)));

		for (int i = 0; i < svm_param.nr_class - 1; i++) {
			model.sv_coef[i] = (double*) (calloc(total_sv, sizeof(double)));
		}

		p = 0;
		for (int i = 0; i < svm_param.nr_class; i++) {
			for (int j = i + 1; j < svm_param.nr_class; j++) {
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = class_count[i];
				int cj = class_count[j];

				int q = nz_start[i];
				int k;
				for (k = 0; k < ci; k++) {
					if (nonzero[si + k]) {
						model.sv_coef[j - 1][q] =
								train_data.label[inputdata_index[si + k]]
										* decisions_k[p].alpha[k];
						q++;
					}
				}

				q = nz_start[j];
				for (k = 0; k < cj; k++) {
					if (nonzero[sj + k]) {
						model.sv_coef[i][q] =
								train_data.label[inputdata_index[sj + k]]
										* decisions_k[p].alpha[ci + k];
						q++;
					}
				}
				p++;
			}
		}

		double percentage_error;
		percentage_error = PredictError(train_data, model);

		logresults(LogfileID, "Classification Errors %5.2f %%\n",
				percentage_error);

		if (cmdOptionExists(argv, argv + argc, "-e")) {
			dataset test_data;
			fileNameTest = getCmdOption(argv, argv + argc, "-e");
			readin_Data(fileNameTest, &test_data, file_gamma, false);
			percentage_error = PredictError(test_data, model);
			logresults(LogfileID, "Prediction Errors %5.2f %%\n",
					percentage_error);
		}

		free(class_count);
		free(inputdata_index);
		free(start);
		free(nonzero);
		for (int i = 0; i < multi_k; i++) {
			free(decisions_k[i].alpha);
		}

		free(decisions_k);
		free(nz_count);
		free(nz_start);

		free(sub_prob->X);
		free(sub_prob->y);
		free(sub_prob->alpha_delta);
		free(sub_prob->alpha_opt_count);
		free(sub_prob->alpha_optimized);
		free(sub_prob->active);

		free(train_data.X);
		free(train_data.label);

		fclose(LogfileID);
		printf("done!\n\n");
	} catch (bad_alloc&) {
		printf(
				"out of memory, you may try \"-m memory size\" to constrain memory usage");
		fclose(LogfileID);
		exit(EXIT_FAILURE);
	} catch (exception const &x) {
		printf("%s\n", x.what());
		fclose(LogfileID);
		exit(EXIT_FAILURE);
	} catch (...) {
		fclose(LogfileID);
		printf("unknown error");
		exit(EXIT_FAILURE);
	}
}

double PredictError(dataset test_problem, svm_model model) {
	int predict_label;
	int incorrect = 0;
	double percentage_error;
	double wall_time;
	chrono::system_clock::time_point start_time = chrono::system_clock::now();

	double *XDot = (double*) (calloc(model.nData, sizeof(double)));

	for (int i = 0; i < model.nData; i++) {
		XDot[i] = selfdotProduct(&model.X[i * svm_param.nFeatures],
				svm_param.nFeatures);
	}

	for (int i = 0; i < test_problem.nData; i++) {
		predict_label = svm_predict(&test_problem.X[i * svm_param.nFeatures],
				model, XDot);
		if (predict_label != test_problem.label[i]) {
			incorrect++;
		}
	}

	percentage_error = 100.0 * incorrect / test_problem.nData;

	wall_time = get_elapsed_time(start_time);
	logresults(LogfileID, "Prediction time : %f\n", wall_time);
	return (percentage_error);
}

void get_WSS(svm_problem *alfpgm_prob, double *alpha, double *y_wss,
		double *alpha_wss, double *minusF, double *H_B_BN, double *G_BN) {
	switch (alfpgm_param.wss_type) {
	case 4:
	case 5:
	case 6:
	case 7:
	case 8:
	default: {
		wss_type_1(alfpgm_prob, alpha, y_wss, alpha_wss, minusF, H_B_BN, G_BN);
	}
		break;
	case 0:
	case 1:
	case 2:
	case 3: {
		wss_type_0(alfpgm_prob, alpha, y_wss, alpha_wss, minusF, H_B_BN, G_BN);
	}
		break;
	case 9: {
		if (alfpgm_param.ndecomp < 20) {
			wss_type_0(alfpgm_prob, alpha, y_wss, alpha_wss, minusF, H_B_BN,
					G_BN);
		} else {
			wss_type_1(alfpgm_prob, alpha, y_wss, alpha_wss, minusF, H_B_BN,
					G_BN);
		}
	}
		break;
	}
}

decision_function SVMSOLVER(svm_problem *alfpgm_prob) {

	decision_function decision_k;
	alfpgm_param.ndecomp = 0;
	alfpgm_param.nWss = 0;
	alfpgm_param.theta = 0.1f;
	alfpgm_param.reqAccuracy = 0;

	alfpgm_param.total_outer_iter = 0;
	alfpgm_param.total_inner_iter = 0;

	double *alpha, *alpha_wss, *y_wss;

	double *grad;

	double *minusF;
	double *H_B_BN;
	double *G_BN;

	alfpgm_prob->beq = 0;
	double norm_g;
	int k, afpgm_exit_flag;

	alfpgm_prob->wss_previous.clear();
	alfpgm_prob->wss_current.clear();

	double wall_time, cpu_time;
	chrono::system_clock::time_point start_time;
	clock_t start_clock, end_clock;

	alfpgm_param.max_pairs = alfpgm_param.npairs;

	if (alfpgm_param.max_pairs >= alfpgm_prob->nData / 2) {
		alfpgm_param.max_pairs = (int) (alfpgm_prob->nData) / 2;
	}

	if (alfpgm_param.max_pairs > svm_param.maxpairs) {
		alfpgm_param.max_pairs = svm_param.maxpairs;
	}

	if (alfpgm_param.max_pairs < svm_param.minpairs) {
		alfpgm_param.max_pairs = svm_param.minpairs;
	}

	if (alfpgm_param.min_alpha_opt == 0) {
		alfpgm_param.min_alpha_opt = min(alfpgm_param.min_alpha_opt_max_val,
				max(alfpgm_param.min_alpha_opt_min_val,
						(alfpgm_prob->nData / alfpgm_param.max_pairs / 20)));
	}

	if (alfpgm_param.max_alpha_opt == 0) {
		alfpgm_param.max_alpha_opt = min(alfpgm_param.max_alpha_opt_max_val,
				max(alfpgm_param.max_alpha_opt_min_val,
						(alfpgm_prob->nData / alfpgm_param.max_pairs / 10)));
	}

	int max_wss = 2 * alfpgm_param.max_pairs;

	logresults(LogfileID, "Gamma: %f\n", svm_param.gamma);
	logresults(LogfileID, "C: %f\n", svm_param.C);
	logresults(LogfileID, "Features: %d\n", svm_param.nFeatures);
	logresults(LogfileID, "mu: %f\n", alfpgm_param.mu);
	logresults(LogfileID, "Max Decomp: %d\n", alfpgm_param.max_decomp);
	logresults(LogfileID, "Max pairs: %d\n", alfpgm_param.max_pairs);
	logresults(LogfileID, "min_alpha_opt: %d\n", alfpgm_param.min_alpha_opt);
	logresults(LogfileID, "max_alpha_opt: %d\n\n", alfpgm_param.max_alpha_opt);

	start_clock = clock();
	start_time = chrono::system_clock::now();

	minusF = (double*) (calloc(alfpgm_prob->nData, sizeof(double)));
	y_wss = (double*) (calloc(max_wss, sizeof(double)));
	alpha = (double*) (calloc(alfpgm_prob->nData, sizeof(double)));
	alpha_wss = (double*) (calloc(max_wss, sizeof(double)));

	H_B_BN = (double*) (calloc(max_wss * alfpgm_prob->nData, sizeof(double)));
	G_BN = (double*) (calloc(max_wss, sizeof(double)));

	grad = (double*) (calloc(max_wss, sizeof(double)));
	double *alpha_differences_vector = (double*) (calloc(max_wss,
			sizeof(double)));
	alfpgm_prob->alpha_opt_count = (int*) realloc(alfpgm_prob->alpha_opt_count,
			sizeof(int) * alfpgm_prob->nData);

	alfpgm_prob->active = (bool*) realloc(alfpgm_prob->active,
			sizeof(bool) * alfpgm_prob->nData);

	alfpgm_prob->alpha_delta = (double*) realloc(alfpgm_prob->alpha_delta,
			sizeof(double) * alfpgm_prob->nData);

	alfpgm_prob->alpha_optimized = (bool*) realloc(alfpgm_prob->alpha_optimized,
			sizeof(bool) * alfpgm_prob->nData);

	alfpgm_prob->K = NULL;
	cache_size = alfpgm_param.cache_size_GB * (1 << 30);
	cache_size -= alfpgm_prob->nData * sizeof(head_t);
	cache_size /= sizeof(double);
	if ((size_t) ((2 + alfpgm_prob->nData * sizeof(head_t) / sizeof(double)))
			> cache_size) {
		cache_size = (2 + alfpgm_prob->nData * sizeof(head_t) / sizeof(double));
	}
	head = (head_t*) ((calloc(alfpgm_prob->nData, sizeof(head_t))));
	lru_head.next = lru_head.prev = &lru_head;

	memcpy(minusF, alfpgm_prob->y, sizeof(double) * alfpgm_prob->nData);
	compose_K_F(alfpgm_prob);

	for (int i = 0; i < alfpgm_prob->nData; i++) {
		alfpgm_prob->alpha_opt_count[i] = 0;
		alfpgm_prob->active[i] = true;
		alfpgm_prob->alpha_optimized[i] = false;
		alfpgm_prob->alpha_delta[i] = svm_param.C;
		alfpgm_prob->all_index.push_back(i);
	}

	sort(alfpgm_prob->all_index.begin(), alfpgm_prob->all_index.end());
	wss_type_0(alfpgm_prob, alpha, y_wss, alpha_wss, minusF, H_B_BN, G_BN);

	bool solve_svm = true;
	double diff_val;
	decision_k.exit_flag = -1;
	if (alfpgm_param.nWss < 2) {
		solve_svm = false;
		decision_k.exit_flag = 4;
	}

	while (solve_svm) {
		afpgm_exit_flag = ALFPGM(alfpgm_prob, y_wss, H_B_BN, alpha_wss, G_BN,
				grad);

		int wss_idx;
		for (int k = 0; k < alfpgm_param.nWss; k++) {
			wss_idx = alfpgm_prob->wss_index[k];
			diff_val = alpha_wss[k] - alpha[wss_idx];
			alpha_differences_vector[k] = diff_val;
			alpha[wss_idx] = alpha_wss[k];
			alfpgm_prob->alpha_delta[wss_idx] = diff_val;
		}

		updateF(minusF, alfpgm_prob, H_B_BN, alpha_differences_vector);

		if (alfpgm_param.verbose > 0) {
			logresults(LogfileID,
					"NumDecompositions #%d  - Total Outer Iterations %d - Total Inner Iterations %d, nWss %d, Gap %f, fpgm exit flag %d\n",
					alfpgm_param.ndecomp, alfpgm_param.total_outer_iter,
					alfpgm_param.total_inner_iter, alfpgm_param.nWss,
					alfpgm_prob->gap, afpgm_exit_flag);
			fflush(LogfileID);
		}
		alfpgm_param.ndecomp++;

		norm_g = 0;
#pragma omp parallel default(none) shared(alfpgm_prob, minusF) reduction(+: norm_g) private(k)
		for (k = 0; k < alfpgm_prob->nData; k++) {
			norm_g += abs(minusF[k]) * abs(minusF[k]);

		}
		norm_g /= alfpgm_prob->numOptimized;

		if ((norm_g < alfpgm_param.accuracy) && (solve_svm)) {
			solve_svm = false;
			decision_k.exit_flag = 0;
		}

		if ((alfpgm_param.ndecomp > alfpgm_param.max_decomp) && (solve_svm)) {
			solve_svm = false;
			decision_k.exit_flag = 1;

		}

		if ((alfpgm_param.total_inner_iter > alfpgm_param.max_total_iterations)
				&& (solve_svm)) {
			solve_svm = false;
			decision_k.exit_flag = 2;

		}

		if (solve_svm) {
			get_WSS(alfpgm_prob, alpha, y_wss, alpha_wss, minusF, H_B_BN, G_BN);
			if (alfpgm_prob->gap < alfpgm_param.accuracy) {
				solve_svm = false;
				decision_k.exit_flag = 0;
			}

			if (alfpgm_param.nWss <= 0) {
				solve_svm = false;
				decision_k.exit_flag = 3;
			}
		}
	}

	end_clock = clock();
	wall_time = get_elapsed_time(start_time);
	cpu_time = (double) (((end_clock - start_clock))) / CLOCKS_PER_SEC;
	logresults(LogfileID, "Training cpu time  : %f\n", cpu_time);
	logresults(LogfileID, "Training wall time : %f\n", wall_time);
	fflush(LogfileID);

	decision_k.alpha = (double*) realloc(decision_k.alpha,
			sizeof(double) * alfpgm_prob->nData);
	memcpy(decision_k.alpha, alpha, alfpgm_prob->nData * sizeof(double));

	decision_k.rho = compute_final_objective_bias(alfpgm_prob, alpha);

	logresults(LogfileID, "WSS Decompositions = %10d\n", alfpgm_param.ndecomp);
	logresults(LogfileID, "Outer Iterations = %10d\n",
			alfpgm_param.total_outer_iter);
	logresults(LogfileID, "Inner Iterations = %10d\n",
			alfpgm_param.total_inner_iter);

	fflush(LogfileID);

	free(minusF);
	free(y_wss);
	free(alpha);
	free(alpha_wss);
	free(H_B_BN);
	free(G_BN);
	free(grad);
	free(alpha_differences_vector);

	for (head_t *h = lru_head.next; h != &lru_head; h = h->next)
		free(h->data);
	free(head);

	return (decision_k);
}

double selfdotProduct(const double *vecA, int n) {
	double dotP = 0;

	dotP = cblas_ddot(n, vecA, 1, vecA, 1);
	return (dotP);
}

int svm_predict(const double *Xi, const svm_model model, double *XDot) {
	int i, k;
	int nr_class = model.nr_class;
	double *kvalue = (double*) (calloc(model.nData, sizeof(double)));
	double *dec_values = (double*) (malloc(
			(svm_param.nr_class * (svm_param.nr_class - 1) / 2)
					* sizeof(double)));

	memcpy(kvalue, XDot, sizeof(double) * model.nData);
	double gamma_xdot = -svm_param.gamma
			* selfdotProduct(Xi, svm_param.nFeatures);
	double a = 2 * svm_param.gamma;
	double b = -svm_param.gamma;
	cblas_dgemv(CblasRowMajor, CblasNoTrans, model.nData, svm_param.nFeatures,
			a, model.X, svm_param.nFeatures, Xi, 1, b, kvalue, 1);

	int *start = (int*) (malloc(svm_param.nr_class * sizeof(int)));
	start[0] = 0;
	for (i = 1; i < nr_class; i++) {
		start[i] = start[i - 1] + model.nSV[i - 1];
	}

	int *vote = (int*) calloc(svm_param.nr_class, sizeof(int));

	int p = 0;
	double sum;
	for (i = 0; i < nr_class; i++) {
		for (int j = i + 1; j < nr_class; j++) {
			sum = 0;
			int si = start[i];
			int sj = start[j];
			int ci = model.nSV[i];
			int cj = model.nSV[j];

			double *coef1 = model.sv_coef[j - 1];
			double *coef2 = model.sv_coef[i];
			for (k = 0; k < ci; k++) {
				sum += coef1[si + k] * exp(gamma_xdot + kvalue[si + k]);
			}

			for (k = 0; k < cj; k++) {
				sum += coef2[sj + k] * exp(gamma_xdot + kvalue[sj + k]);
			}

			sum -= model.rho[p];

			dec_values[p] = sum;
			if (dec_values[p] >= 0)
				++vote[i];
			else
				++vote[j];
			p++;
		}
	}

	int vote_max_idx = 0;
	for (i = 1; i < nr_class; i++) {
		if (vote[i] > vote[vote_max_idx]) {
			vote_max_idx = i;
		}
	}

	free(dec_values);
	free(kvalue);
	free(start);
	free(vote);

	return model.labels[vote_max_idx];
}

double compute_final_objective_bias(svm_problem *alfpgm_prob, double *alpha) {
	double sumAlpha = 0;
	double alphaQalpha = 0;
	double alphaQ = 0;
	int i, j;
	double b = 0;

	double *Xi, alphaKij;

	int nsv = 0, nBsv = 0;
	for (i = 0; i < alfpgm_prob->nData; i++) {
		if (!is_lower_bound_alpha(alpha[i])) {
			nsv++;
		}
		if (is_upper_bound_alpha(alpha[i])) {
			nBsv++;
		}
	}

	logresults(LogfileID,
			"Number of Support Vectors : %d\nNumber of Bounded Support Vectors : %d\nNumber of Optimized Alphas : %d\n",
			nsv, nBsv, alfpgm_prob->numOptimized);

	double *Ki = (double*) (calloc(alfpgm_prob->nData, sizeof(double)));

	{
		for (i = 0; i < alfpgm_prob->nData; i++) {
			sumAlpha += alpha[i];
			if (!is_lower_bound_alpha(alpha[i])) {
				alphaQ = 0;
				Xi = &alfpgm_prob->X[i * svm_param.nFeatures];

				double xdot = alfpgm_prob->Xdot[i];
				cblas_dgemv(CblasRowMajor, CblasNoTrans, alfpgm_prob->nData,
						svm_param.nFeatures, 1.0, alfpgm_prob->X,
						svm_param.nFeatures, Xi, 1, 0, Ki, 1);
#pragma omp parallel for default(none) shared(xdot, alpha, svm_param, Ki, alfpgm_prob) private(j, alphaKij) schedule(guided) reduction(+ : alphaQ, alphaQalpha, sumAlpha, b)
				for (j = 0; j < alfpgm_prob->nData; j++) {
					Ki[j] =
							exp(
									-svm_param.gamma
											* (xdot + alfpgm_prob->Xdot[j]
													- 2 * Ki[j]));
					alphaKij = alpha[j] * Ki[j];
					alphaQ += alphaKij * alfpgm_prob->y[j];
					b += alphaKij * alfpgm_prob->y[j];

				}
				alphaQalpha += alpha[i] * alphaQ * alfpgm_prob->y[i];
				b += -alfpgm_prob->y[i];

			}
		}
	}
	free(Ki);

	b = b / nsv;

	double objective_value = -sumAlpha + .5f * alphaQalpha;
	logresults(LogfileID, "Objective Value %f, bias %f\n", objective_value, b);

	return (b);
}

bool is_lower_bound_alpha(double alpha_i) {
	return (alpha_i < alfpgm_param.epsilon);
}

bool is_upper_bound_alpha(double alpha_i) {
	return (fabs(alpha_i - svm_param.C) < alfpgm_param.epsilon);
}

double get_elapsed_time(const chrono::system_clock::time_point start_time) {
	chrono::system_clock::time_point stop_time = chrono::system_clock::now();
	chrono::duration<double> wall_time = stop_time - start_time;
	return (wall_time.count());
}

void updateF(double *minusF, svm_problem *alfpgm_prob, const double *H_B_BN,
		const double *alpha_differences_vector) {

	int i;

	double *alphaHVec = (double*) (calloc(alfpgm_prob->nData, sizeof(double)));
	cblas_dgemv(CblasRowMajor, CblasTrans, alfpgm_param.nWss,
			alfpgm_prob->nData, 1, H_B_BN, alfpgm_prob->nData,
			alpha_differences_vector, 1, 0, alphaHVec, 1);

#pragma omp parallel private(i) default(none) shared(minusF, alphaHVec, alfpgm_prob)
	{
#pragma omp for  schedule(guided)
		for (i = 0; i < alfpgm_prob->nData; i++) {
			minusF[alfpgm_prob->wss_index[i]] -=
					alfpgm_prob->y[alfpgm_prob->wss_index[i]] * alphaHVec[i];
		}
	}

	free(alphaHVec);

}

double rbf_dist_vector(const double *Xi, const double *Xj) {
	double normsquared = 0, alpha_diff;
	for (int k = 0; k < svm_param.nFeatures; k++) {
		alpha_diff = Xi[k] - Xj[k];
		normsquared += alpha_diff * alpha_diff;
	}
	return (exp(-svm_param.gamma * normsquared));
}

double getK_ij(svm_problem *alfpgm_prob, double *Xi, int j) {
	double Kij;
	Kij = rbf_dist_vector(Xi, &alfpgm_prob->X[j * svm_param.nFeatures]);
	return (Kij);
}

void getK_i(svm_problem *alfpgm_prob, double *K_i, int i) {
	double *Ktemp;
	int j;
	int cached_length = get_data(alfpgm_prob, i, &Ktemp);
	if (cached_length < alfpgm_prob->nData) {

		double a = 2 * svm_param.gamma;
		double b = -svm_param.gamma;

		double *Xi;
		Xi = &alfpgm_prob->X[i * svm_param.nFeatures];

		double *K = (double*) (calloc(alfpgm_prob->nData, sizeof(double)));
		memcpy(K, alfpgm_prob->Xdot, sizeof(double) * alfpgm_prob->nData);
		double gamma_xdot = -svm_param.gamma * alfpgm_prob->Xdot[i];
		cblas_dgemv(CblasRowMajor, CblasNoTrans, alfpgm_prob->nData,
				svm_param.nFeatures, a, alfpgm_prob->X, svm_param.nFeatures, Xi,
				1, b, K, 1);
#pragma omp parallel for default(none) shared(gamma_xdot, svm_param, Ktemp, K, alfpgm_prob) private(j) schedule(guided)
		for (j = 0; j < alfpgm_prob->nData; j++) {
			Ktemp[j] = exp(gamma_xdot + K[j]);
		}
		free(K);

	}
	memcpy(K_i, Ktemp, sizeof(double) * alfpgm_prob->nData);
}

void getQ_i(svm_problem *alfpgm_prob, double *Q_i, int i) {
	int j;
	getK_i(alfpgm_prob, Q_i, i);
	double yi = alfpgm_prob->y[i];
#pragma omp parallel for default(none) shared(alfpgm_prob, Q_i, yi) private(j) schedule(guided)
	for (j = 0; j < alfpgm_prob->nData; j++) {
		Q_i[j] = yi * alfpgm_prob->y[j] * Q_i[j];
	}
}

int get_second_order_mvp_for_i1(double *Ki, int i_up,
		vector<pair<double, int>> &Ilow, vector<int> &wss_list,
		double *minusF) {

	double a, b;
	int mvp_idx = -1;
	double obj_val;
	double obj_min = DBL_MAX;
	double minusFi = minusF[i_up];
	double Kij;
	int j_idx;
	int counter = 0;

	int min_check = min(svm_param.min_check_max,
			ceil(Ilow.size() * svm_param.min_check_frac));

	for (auto &&J : Ilow) {
		j_idx = J.second;
		counter++;
		if (j_idx != i_up) {
			if (std::find(wss_list.begin(), wss_list.end(), j_idx)
					== wss_list.end()) {
				b = minusFi - minusF[j_idx];
				switch (alfpgm_param.wss_type) {
				case 4: {

					if (b > 0) {
						Kij = Ki[j_idx];
						a = 2.0 - 2.0 * Kij;
						a = fmax(a, alfpgm_param.epsilon);
						obj_val = -(b * b) / a;

						if ((obj_val < obj_min)) {
							mvp_idx = j_idx;
							obj_min = obj_val;
						}
					}
					if (counter >= min_check) {
						return (mvp_idx);
					}
					break;
				}
				default: {
					if (b > alfpgm_param.accuracy) {
						Kij = Ki[j_idx];
						a = 2.0 - 2.0 * Kij;
						a = fmax(a, alfpgm_param.epsilon);
						obj_val = -(b * b) / a;
						if ((obj_val < obj_min)) {
							mvp_idx = j_idx;
							obj_min = obj_val;
						}
						if ((alfpgm_param.wss_type > 4)
								&& (counter >= min_check) && (mvp_idx != -1)) {
							return (mvp_idx);
						}

					}
					if (counter >= min_check) {
						return (mvp_idx);
					}
					break;
				}
				}
			}
		}

	}

	return (mvp_idx);

}

int get_second_order_mvp_for_i2(svm_problem *alfpgm_prob, double *Xi, int i_up,
		vector<pair<double, int>> &Ilow, vector<int> &wss_list,
		double *minusF) {

	double a, b;
	int mvp_idx = -1;
	double obj_val;
	double obj_min = DBL_MAX;
	double minusFi = minusF[i_up];
	double Kij;
	int j_idx;
	int counter = 0;

	int min_check = min(svm_param.min_check_max,
			ceil(Ilow.size() * svm_param.min_check_frac));

	for (auto &&J : Ilow) {
		j_idx = J.second;
		counter++;
		if (j_idx != i_up) {
			if (std::find(wss_list.begin(), wss_list.end(), j_idx)
					== wss_list.end()) {

				b = minusFi - minusF[j_idx];
				switch (alfpgm_param.wss_type) {
				case 4: {
					if (b > 0) {
						Kij = getK_ij(alfpgm_prob, Xi, j_idx);

						a = 2.0 - 2.0 * Kij;
						a = fmax(a, alfpgm_param.epsilon);
						obj_val = -(b * b) / a;

						if ((obj_val < obj_min)) {
							mvp_idx = j_idx;
							obj_min = obj_val;

						}
					}
					if (counter >= min_check) {
						return (mvp_idx);
					}
					break;
				}
				default: {
					if (b > alfpgm_param.accuracy) {
						Kij = getK_ij(alfpgm_prob, Xi, j_idx);

						a = 2.0 - 2.0 * Kij;
						a = fmax(a, alfpgm_param.epsilon);
						obj_val = -(b * b) / a;

						if ((obj_val < obj_min)) {
							mvp_idx = j_idx;
							obj_min = obj_val;
						}
						if ((alfpgm_param.wss_type > 4)
								&& (counter >= min_check) && (mvp_idx != -1)) {
							return (mvp_idx);
						}
					}
					if (counter >= min_check) {
						return (mvp_idx);
					}
					break;
				}
				}
			}
		}
	}

	return (mvp_idx);

}

void compose_K_F(svm_problem *alfpgm_prob) {
	chrono::system_clock::time_point start_time;
	alfpgm_prob->Xdot = (double*) (realloc(alfpgm_prob->Xdot,
			sizeof(double) * alfpgm_prob->nData));

	for (int i = 0; i < alfpgm_prob->nData; i++) {
		alfpgm_prob->Xdot[i] = cblas_ddot(svm_param.nFeatures,
				&alfpgm_prob->X[i * svm_param.nFeatures], 1,
				&alfpgm_prob->X[i * svm_param.nFeatures], 1);
	}

}

void Add_Iup_Ilow(int i, svm_problem *alfpgm_prob, double *alpha,
		vector<pair<double, int> > &Iup, double *minusF,
		vector<pair<double, int> > &Ilow) {

	if (((alfpgm_prob->y[i] > 0) && !is_upper_bound_alpha(alpha[i]))
			|| ((alfpgm_prob->y[i] < 0) && !is_lower_bound_alpha(alpha[i]))) {
		Iup.push_back(make_pair(minusF[i], i));
	}
	if (((alfpgm_prob->y[i] > 0) && !is_lower_bound_alpha(alpha[i]))
			|| ((alfpgm_prob->y[i] < 0) && !is_upper_bound_alpha(alpha[i]))) {
		Ilow.push_back(make_pair(minusF[i], i));
	}
}

void get_wss2_Iup_Ilow(svm_problem *alfpgm_prob, double *alpha,
		vector<int> active_list, vector<pair<double, int>> &Iup, double *minusF,
		vector<pair<double, int>> &Ilow) {
	for (auto &&i : active_list) {
		if (alfpgm_prob->alpha_opt_count[i] < alfpgm_param.max_alpha_opt) {
			if (alfpgm_prob->alpha_opt_count[i] >= alfpgm_param.min_alpha_opt) {

				if ((alfpgm_prob->alpha_delta[i]
						< alfpgm_param.alpha_delta_threshold)
						|| ((is_lower_bound_alpha(alpha[i])
								|| is_upper_bound_alpha(alpha[i])))) {
					alfpgm_prob->alpha_opt_count[i] =
							alfpgm_param.max_alpha_opt;
					continue;
				}
			}
			Add_Iup_Ilow(i, alfpgm_prob, alpha, Iup, minusF, Ilow);
		}
	}
}

void find_Iup_Ilow(svm_problem *alfpgm_prob, vector<pair<double, int>> &Iup,
		vector<pair<double, int>> &Ilow, double *alpha, double *minusF) {

	Iup.clear();
	Ilow.clear();
	alfpgm_prob->gap = 0;

	vector<int> active_list;

	if (alfpgm_prob->do_not_use_same) {
		set_difference(alfpgm_prob->all_index.begin(),
				alfpgm_prob->all_index.end(), alfpgm_prob->wss_current.begin(),
				alfpgm_prob->wss_current.end(),
				inserter(active_list, active_list.begin()));
	} else {
		active_list = alfpgm_prob->all_index;
	}

	switch (alfpgm_param.wss_type) {

	case 0:
	case 4:
	case 5: // wss2 limited

	default: {
		for (auto &&i : active_list) {
			Add_Iup_Ilow(i, alfpgm_prob, alpha, Iup, minusF, Ilow);
		}
	}
		break;
	case 1:
	case 6:

	{ // MVP Limited by min

		for (auto &&i : active_list) {
			if (alfpgm_prob->active[i]) {
				if ((alfpgm_prob->alpha_opt_count[i]
						> alfpgm_param.min_alpha_opt)
						&& (is_lower_bound_alpha(alpha[i])
								|| is_upper_bound_alpha(alpha[i])
								|| (alfpgm_prob->alpha_delta[i]
										< alfpgm_param.alpha_delta_threshold))) {
					alfpgm_prob->active[i] = false;
					continue;

				}

				Add_Iup_Ilow(i, alfpgm_prob, alpha, Iup, minusF, Ilow);

			}
		}

	}
		break;
	case 2:
	case 7:
	case 9: { // MVP Limited by min and max

		for (auto &&i : active_list) {

			if (alfpgm_prob->active[i]) {
				if (alfpgm_prob->alpha_opt_count[i]
						< alfpgm_param.max_alpha_opt)

						{
					if ((alfpgm_prob->alpha_opt_count[i]
							> alfpgm_param.min_alpha_opt)
							&& (is_lower_bound_alpha(alpha[i])
									|| is_upper_bound_alpha(alpha[i])
									|| (alfpgm_prob->alpha_delta[i]
											< alfpgm_param.alpha_delta_threshold))) {
						alfpgm_prob->alpha_opt_count[i] =
								alfpgm_param.max_alpha_opt;
						alfpgm_prob->active[i] = false;
						continue;

					}

					Add_Iup_Ilow(i, alfpgm_prob, alpha, Iup, minusF, Ilow);
				} else {
					alfpgm_prob->active[i] = false;
				}
			}
		}

	}

		break;
	case 3:
	case 8: {

		// MVP Limited by min and max
		find_min_count(alfpgm_prob);

		bool do_loop = true;

		while (do_loop) {
			get_wss2_Iup_Ilow(alfpgm_prob, alpha, active_list, Iup, minusF,
					Ilow);

			if (((Iup.size() == 0) || (Ilow.size() == 0))
					&& (alfpgm_prob->min_count < alfpgm_param.max_alpha_opt)) {
				alfpgm_prob->min_count++;
			} else {
				do_loop = false;
			}
		}

	}
		break;
	}
	sort(Iup.begin(), Iup.end(), sortbyfirstdesc);
	sort(Ilow.begin(), Ilow.end(), sortbyfirstcasc);

	if (alfpgm_param.verbose >= 2) {
		logresults(LogfileID, "n_i : %d  n_j : %d\n", (int )Iup.size(),
				(int )Ilow.size());
	}

}

void find_min_count(svm_problem *alfpgm_prob) {
	int min_count = alfpgm_prob->alpha_opt_count[0];
#pragma omp parallel for default(none) shared(alfpgm_prob) reduction(min: min_count)
	for (int i = 0; i < alfpgm_prob->nData; i++) {
		if (alfpgm_prob->alpha_opt_count[i] < min_count) {
			min_count = alfpgm_prob->alpha_opt_count[i];
		}
	}

	alfpgm_prob->min_count = min_count;
}

bool check_all_labels(svm_problem *alfpgm_prob, vector<int> &idx) {
	int plus_label = 0;
	int negative_label = 0;
	for (auto &&i : idx) {
		if (alfpgm_prob->y[i] > 0) {
			plus_label++;
		} else if (alfpgm_prob->y[i] < 0) {
			negative_label++;
		}
		if (plus_label > 0 && negative_label > 0) {
			return (true);
		}
	}

	return (false);
}

void wss_type_1(svm_problem *alfpgm_prob, double *alpha, double *y_wss,
		double *alpha_wss, double *minusF, double *H_B_BN, double *G_BN) {

	chrono::system_clock::time_point wss_start_time =
			chrono::system_clock::now();
	vector<pair<double, int>> Iup;

	vector<int> other_alpha_indexes;
	vector<pair<double, int>> Ilow;

	vector<int> wss_list;
	vector<int> wss_new;

	alfpgm_param.nWss = 0;
	int Itmp, Jtmp;
	int nB = 0;

	alfpgm_prob->wss_index.clear();

	double *H_B_BN_temp = (double*) (calloc(
			2 * alfpgm_param.max_pairs * alfpgm_prob->nData, sizeof(double)));

	double *XXi;

	find_Iup_Ilow(alfpgm_prob, Iup, Ilow, alpha, minusF);

	if ((Iup.size() > 0) && (Ilow.size() > 0)) {

		//% Most violating pairs - Keerti and Gilbert (2002)
		//% Select the second alpha (j)

		for (auto &&Iupval : Iup) {
			if (nB < (2 * alfpgm_param.max_pairs)) {
				Itmp = Iupval.second;
				if (std::find(wss_list.begin(), wss_list.end(), Itmp)
						== wss_list.end()) {
					XXi = &alfpgm_prob->X[Itmp * svm_param.nFeatures];
					Jtmp = get_second_order_mvp_for_i2(alfpgm_prob, XXi, Itmp,
							Ilow, wss_list, minusF);

					if (Jtmp != -1) {
						assign_Itmp_Jtmp(alfpgm_prob, alpha, y_wss, alpha_wss,
								H_B_BN_temp, nB, Itmp, Jtmp, minusF);

						wss_list.push_back(Itmp);
						wss_list.push_back(Jtmp);
						nB += 2;
					}
				}
			} else {
				break;
			}
		}

		if (nB > 0) {

			alfpgm_prob->beq = 0;

			sort(wss_list.begin(), wss_list.end());
			alfpgm_prob->wss_previous = alfpgm_prob->wss_current;
			alfpgm_prob->wss_current = wss_list;

			set_difference(alfpgm_prob->all_index.begin(),
					alfpgm_prob->all_index.end(), wss_list.begin(),
					wss_list.end(),
					inserter(other_alpha_indexes, other_alpha_indexes.begin()));

			set_difference(alfpgm_prob->wss_current.begin(),
					alfpgm_prob->wss_current.end(),
					alfpgm_prob->wss_previous.begin(),
					alfpgm_prob->wss_previous.end(),
					inserter(wss_new, wss_new.begin()));

			for (auto &&i : other_alpha_indexes) {
				{
					alfpgm_prob->wss_index.push_back(i);
					alfpgm_prob->beq -= alfpgm_prob->y[i] * alpha[i];
				}
			}

			double tmp = 0;
			for (int i = 0; i < nB; i++) {
				for (int j = 0; j < nB; j++) {
					H_B_BN[i * alfpgm_prob->nData + j] = H_B_BN_temp[i
							* alfpgm_prob->nData + alfpgm_prob->wss_index[j]];
				}

				//	G_BN = (H_BN * alpha_N) - I;
				double alphaHBN = -1;
				for (int j = nB; j < alfpgm_prob->nData; j++) {
					tmp = H_B_BN_temp[i * alfpgm_prob->nData
							+ alfpgm_prob->wss_index[j]];
					H_B_BN[i * alfpgm_prob->nData + j] = tmp;
					alphaHBN += alpha[alfpgm_prob->wss_index[j]] * tmp;
				}

				G_BN[i] = alphaHBN;
			}

			alfpgm_prob->numOptimized = 0;

			for (int j = 0; j < alfpgm_prob->nData; j++) {
				if (alfpgm_prob->alpha_optimized[j]) {
					alfpgm_prob->numOptimized++;
				}
			}
		}
		alfpgm_param.nWss = nB;
	}

	else {
		alfpgm_param.nWss = 0;
	}

	free(H_B_BN_temp);
	double wall_time = get_elapsed_time(wss_start_time);
	alfpgm_prob->same_wss = wss_new.size() == 0;
	if ((alfpgm_param.wss_type != 0) && (alfpgm_param.wss_type != 4)) {
		if (alfpgm_prob->same_wss & !alfpgm_prob->do_not_use_same) {
			printf("Same selected\n");
			alfpgm_prob->do_not_use_same = true;
			wss_type_1(alfpgm_prob, alpha, y_wss, alpha_wss, minusF, H_B_BN,
					G_BN);
		} else {
			alfpgm_prob->same_wss = false;
		}
	}
	if (alfpgm_param.verbose >= 1) {
		logresults(LogfileID,
				"Decomposition %d, WSS Time : %f nWss : %d, Same WSS? : %d\n",
				alfpgm_param.ndecomp, wall_time, alfpgm_param.nWss,
				alfpgm_prob->same_wss);
	}

}

void assign_Itmp_Jtmp(svm_problem *alfpgm_prob, double *alpha, double *y_wss,
		double *alpha_wss, double *H_B_BN_temp, int nB, int Itmp, int Jtmp,
		double *minusF) {

	alfpgm_prob->wss_index.push_back(Itmp);
	y_wss[nB] = alfpgm_prob->y[Itmp];
	alpha_wss[nB] = alpha[Itmp];
	alfpgm_prob->alpha_opt_count[Itmp]++;
	alfpgm_prob->alpha_opt_count[Jtmp]++;
	alfpgm_prob->alpha_optimized[Itmp] = true;
	alfpgm_prob->alpha_optimized[Jtmp] = true;
	alfpgm_prob->wss_index.push_back(Jtmp);
	y_wss[(nB + 1)] = alfpgm_prob->y[Jtmp];
	alpha_wss[(nB + 1)] = alpha[Jtmp];
	alfpgm_prob->gap = max(alfpgm_prob->gap, minusF[Itmp] - minusF[Jtmp]);

	getQ_i(alfpgm_prob, &H_B_BN_temp[nB * alfpgm_prob->nData], Itmp);
	getQ_i(alfpgm_prob, &H_B_BN_temp[(nB + 1) * alfpgm_prob->nData], Jtmp);

}

bool sortbyfirstdesc(const pair<double, int> &a, const pair<double, int> &b) {
	return a.first > b.first;
}

bool sortbyfirstcasc(const pair<double, int> &a, const pair<double, int> &b) {
	return a.first < b.first;
}

int fast_get_mvp(int Itmp, vector<pair<double, int>> Ilow, double minusFi,
		vector<int> wss_list) {

	for (auto &&j : Ilow) { // forwarding reference, auto type deduction
		if (Itmp != j.second) {
			if (std::find(wss_list.begin(), wss_list.end(), j.second)
					== wss_list.end()) {

				switch (alfpgm_param.wss_type) {
				case 0:
				case 4: {
					if (minusFi >= j.first) {
						return ((int) j.second);
					} else {
						return (-1);
					}
					break;
				}

				default: {

					if ((minusFi - j.first) >= alfpgm_param.accuracy) {
						return ((int) j.second);
					} else {
						return (-1);
					}
					break;
				}
				}
			}
		}
	}

	return (-1);
}

void wss_type_0(svm_problem *alfpgm_prob, double *alpha, double *y_wss,
		double *alpha_wss, double *minusF, double *H_B_BN, double *G_BN) {

	chrono::system_clock::time_point wss_start_time =
			chrono::system_clock::now();
	vector<pair<double, int>> Iup;

	vector<int> other_alpha_indexes;
	vector<pair<double, int>> Ilow;

	vector<int> wss_list;
	vector<int> wss_new;

	alfpgm_param.nWss = 0;
	int Itmp, Jtmp;

	double minusFi;
	alfpgm_prob->wss_index.clear();

	double *H_B_BN_temp = (double*) (calloc(
			2 * alfpgm_param.max_pairs * alfpgm_prob->nData, sizeof(double)));
	find_Iup_Ilow(alfpgm_prob, Iup, Ilow, alpha, minusF);

	if ((Iup.size() > 0) && (Ilow.size() > 0)) {
		int nB = 0;

		//% Most violating pairs - Keerti and Gilbert (2002)
		//% Select the second alpha (j)

		for (auto &&Iupval : Iup) {
			if (nB < (2 * alfpgm_param.max_pairs)) {
				Itmp = Iupval.second;
				if (std::find(wss_list.begin(), wss_list.end(), Itmp)
						== wss_list.end()) {
					minusFi = minusF[Itmp];
					Jtmp = fast_get_mvp(Itmp, Ilow, minusFi, wss_list);

					if (Jtmp != -1) {
						assign_Itmp_Jtmp(alfpgm_prob, alpha, y_wss, alpha_wss,
								H_B_BN_temp, nB, Itmp, Jtmp, minusF);

						wss_list.push_back(Itmp);
						wss_list.push_back(Jtmp);
						nB += 2;
					}
				}
			} else {
				break;

			}
		}

		if (nB > 0) {

			alfpgm_prob->beq = 0;
			sort(wss_list.begin(), wss_list.end());
			alfpgm_prob->wss_previous = alfpgm_prob->wss_current;
			alfpgm_prob->wss_current = wss_list;

			set_difference(alfpgm_prob->all_index.begin(),
					alfpgm_prob->all_index.end(), wss_list.begin(),
					wss_list.end(),
					inserter(other_alpha_indexes, other_alpha_indexes.begin()));

			set_difference(alfpgm_prob->wss_current.begin(),
					alfpgm_prob->wss_current.end(),
					alfpgm_prob->wss_previous.begin(),
					alfpgm_prob->wss_previous.end(),
					inserter(wss_new, wss_new.begin()));

			for (auto &&i : other_alpha_indexes) {
				{
					alfpgm_prob->wss_index.push_back(i);
					alfpgm_prob->beq -= alfpgm_prob->y[i] * alpha[i];
				}
			}

			double tmp = 0;
			for (int i = 0; i < nB; i++) {
				for (int j = 0; j < nB; j++) {
					H_B_BN[i * alfpgm_prob->nData + j] = H_B_BN_temp[i
							* alfpgm_prob->nData + alfpgm_prob->wss_index[j]];
				}

				//	G_BN = (H_BN * alpha_N) - I;
				double alphaHBN = -1;
				for (int j = nB; j < alfpgm_prob->nData; j++) {
					tmp = H_B_BN_temp[i * alfpgm_prob->nData
							+ alfpgm_prob->wss_index[j]];
					H_B_BN[i * alfpgm_prob->nData + j] = tmp;
					alphaHBN += alpha[alfpgm_prob->wss_index[j]] * tmp;
				}

				G_BN[i] = alphaHBN;
			}

			alfpgm_prob->numOptimized = 0;

			for (int j = 0; j < alfpgm_prob->nData; j++) {
				if (alfpgm_prob->alpha_optimized[j]) {
					alfpgm_prob->numOptimized++;
				}
			}
		}
		alfpgm_param.nWss = nB;
	}

	free(H_B_BN_temp);
	double wall_time = get_elapsed_time(wss_start_time);

	alfpgm_prob->same_wss = wss_new.size() == 0;
	if ((alfpgm_param.wss_type != 0) && (alfpgm_param.wss_type != 4)) {
		if (alfpgm_prob->same_wss & !alfpgm_prob->do_not_use_same) {
			printf("Same selected\n");
			alfpgm_prob->do_not_use_same = true;
			wss_type_0(alfpgm_prob, alpha, y_wss, alpha_wss, minusF, H_B_BN,
					G_BN);
		} else {
			alfpgm_prob->same_wss = false;
		}
	}

	if (alfpgm_param.verbose >= 1) {
		logresults(LogfileID,
				"Decomposition %d, WSS Time : %f nWss : %d, Same WSS? : %d\n",
				alfpgm_param.ndecomp, wall_time, alfpgm_param.nWss,
				alfpgm_prob->same_wss);
	}

}

void readin_Data(const char *file_name, dataset *load_data, bool file_gamma,
		bool isTraining) {
	string line; /* string to hold each line */
	ifstream f(file_name); /* open file */

	if (!f.is_open()) { /* validate file open for reading */
		perror(("error while opening file " + string(file_name)).c_str());
	}

	int row, col;
	string val; /* string to hold value */
	getline(f, val);
	load_data->nData = stoi(val);

	if (isTraining) {
		getline(f, val);
		svm_param.nFeatures = stoi(val);
		if (file_gamma) {
			getline(f, val);
			svm_param.gamma = stof(val);
		}

		logresults(LogfileID,
				"Training File %s\nTraining Data (Rows %d, Features %d) Gamma: %f\r\n",
				file_name, load_data->nData, svm_param.nFeatures,
				svm_param.gamma);
	} else {
		getline(f, val); //Skip nFeatures
		int nFeaturesTest = stoi(val);
		if (svm_param.nFeatures != nFeaturesTest) {
			printf(
					"Test data (%d) and Train Data (%d) do not have same feature set\n",
					svm_param.nFeatures, nFeaturesTest);
			exit(-1);
		}
		logresults(LogfileID,
				"Test File %s\nTest Data (Rows %d, Features %d)\r\n", file_name,
				load_data->nData, nFeaturesTest);
	}

	load_data->X = (double*) calloc(load_data->nData * svm_param.nFeatures,
			sizeof(double));

	load_data->label = (int*) calloc(load_data->nData, sizeof(int));
	row = 0;
	col = -1;
	while (row < load_data->nData) {
		getline(f, line);
		stringstream s(line);
		while (getline(s, val, ',') && (col < svm_param.nFeatures)) {
			if (col < 0) {
				load_data->label[row] = stoi(val);
			} else {
				load_data->X[row * svm_param.nFeatures + col] = stof(val);
			}
			col++;
		}

		col = -1;
		row++;
	}
	f.close();
}

bool cmdOptionExists(char **begin, char **end, const string &option) {
	return find(begin, end, option) != end;
}

char* getCmdOption(char **begin, char **end, const string &option) {
	char **itr = find(begin, end, option);
	if (itr != end && ++itr != end) {
		return *itr;
	}
	return 0;
}

void parse_arguments(int argc, char **argv) {

	if (cmdOptionExists(argv, argv + argc, "-c")) {
		svm_param.C = atof(getCmdOption(argv, argv + argc, "-c"));
	}
	if (cmdOptionExists(argv, argv + argc, "-m")) {
		alfpgm_param.cache_size_GB = atol(
				getCmdOption(argv, argv + argc, "-m"));
	}

	if (cmdOptionExists(argv, argv + argc, "-b")) {
		svm_param.min_check_frac = (int) atol(
				getCmdOption(argv, argv + argc, "-b"));
	}

	if (cmdOptionExists(argv, argv + argc, "-t")) {
		alfpgm_param.accuracy = atof(getCmdOption(argv, argv + argc, "-t"));
		logresults(LogfileID, "accuracy %f \n", alfpgm_param.accuracy);
	}

	if (cmdOptionExists(argv, argv + argc, "-n")) {
		alfpgm_param.alfpgm_accuracy = atof(
				getCmdOption(argv, argv + argc, "-n"));
		logresults(LogfileID, "nral_accuracy %f \n",
				alfpgm_param.alfpgm_accuracy);
	}

	if (cmdOptionExists(argv, argv + argc, "-h")) {
		alfpgm_param.theta = atof(getCmdOption(argv, argv + argc, "-h"));
	}

	if (cmdOptionExists(argv, argv + argc, "-d")) {
		alfpgm_param.max_decomp = atol(getCmdOption(argv, argv + argc, "-d"));
	}

	if (cmdOptionExists(argv, argv + argc, "-s")) {
		alfpgm_param.min_alpha_opt = atol(
				getCmdOption(argv, argv + argc, "-s"));
	}

	if (cmdOptionExists(argv, argv + argc, "-r")) {
		alfpgm_param.max_alpha_opt = atol(
				getCmdOption(argv, argv + argc, "-r"));
	}

	if (cmdOptionExists(argv, argv + argc, "-p")) {
		alfpgm_param.npairs = (int) atol(getCmdOption(argv, argv + argc, "-p"));
	}

	if (cmdOptionExists(argv, argv + argc, "-o")) {
		svm_param.min_check_max = (int) atol(
				getCmdOption(argv, argv + argc, "-o"));
	} else {
		svm_param.min_check_max = alfpgm_param.npairs;
	}

	if (cmdOptionExists(argv, argv + argc, "-v")) {
		alfpgm_param.verbose = (int) (atol(
				getCmdOption(argv, argv + argc, "-v")));
	}

	if (cmdOptionExists(argv, argv + argc, "-g")) {
		svm_param.gamma = (atof(getCmdOption(argv, argv + argc, "-g")));
		logresults(LogfileID, "Gamma: %f\r\n", svm_param.gamma);
	}

	if (cmdOptionExists(argv, argv + argc, "-x")) {
		alfpgm_param.fista_type = int(
				atol(getCmdOption(argv, argv + argc, "-x")));
	}

	if (cmdOptionExists(argv, argv + argc, "-u")) {
		alfpgm_param.mu = atof(getCmdOption(argv, argv + argc, "-u"));
	}

	if (cmdOptionExists(argv, argv + argc, "-w")) {
		alfpgm_param.wss_type = (int) (atol(
				getCmdOption(argv, argv + argc, "-w")));
		if (alfpgm_param.wss_type > 9) {
			alfpgm_param.wss_type = 0;
		}
	}
}

void lru_delete(head_t *h) {
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void lru_insert(head_t *h) {
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int get_data(svm_problem *alfpgm_prob, const int index, double **data) {

	head_t *h = &head[index];
	int cached_length = h->len;
	if (h->len > 0) {
		lru_delete(h);
	} else {
		while (cache_size < (size_t) alfpgm_prob->nData) {
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			cache_size += old->len;
			old->data = 0;
			old->len = 0;
		}
		h->data = (double*) realloc(h->data,
				sizeof(double) * alfpgm_prob->nData);
		h->len = alfpgm_prob->nData;
		cache_size -= alfpgm_prob->nData;
	}
	lru_insert(h);
	*data = h->data;
	return (cached_length);
}

void swapint(int &a, int &b) {
	int c = a;
	a = b;
	b = c;
}

int svm_group_classes(dataset *train_data, int **label_ret, int **start_ret,
		int **count_ret, int *perm) {

	int *label = (int*) (malloc(svm_param.max_nr_class * sizeof(int)));
	int *counter = (int*) (malloc(svm_param.max_nr_class * sizeof(int)));
	int *data_label = (int*) calloc(train_data->nData, sizeof(int));
	int i, j;

	int nr_class = 0;

	for (i = 0; i < train_data->nData; i++) {

		int this_label = train_data->label[i];
		for (j = 0; j < nr_class; j++) {
			if (this_label == label[j]) {
				++counter[j];
				break;
			}
		}

		data_label[i] = j;
		if (j == nr_class) {
			if (nr_class == svm_param.max_nr_class) {
				svm_param.max_nr_class *= 2;
				label = (int*) realloc(label,
						svm_param.max_nr_class * sizeof(int));
				counter = (int*) realloc(counter,
						svm_param.max_nr_class * sizeof(int));
			}
			label[nr_class] = this_label;
			counter[nr_class] = 1;
			++nr_class;
		}
	}

	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.

	if (nr_class == 2 && label[0] == -1 && label[1] == 1) {
		swapint(label[0], label[1]);
		swapint(counter[0], counter[1]);
		for (i = 0; i < train_data->nData; i++) {
			if (data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = (int*) (malloc(svm_param.max_nr_class * sizeof(int)));
	start[0] = 0;
	for (i = 1; i < nr_class; i++) {
		start[i] = start[i - 1] + counter[i - 1];
	}

	for (i = 0; i < train_data->nData; i++) {
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}

	start[0] = 0;
	for (i = 1; i < nr_class; i++) {
		start[i] = start[i - 1] + counter[i - 1];
	}

	*label_ret = label;
	*start_ret = start;
	*count_ret = counter;
	free(data_label);

	return (nr_class);
}

