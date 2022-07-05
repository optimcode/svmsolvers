#include "fista.h"

using namespace std;

int ALFPGM(svm_problem *alfpgm_prob, double *y_wss, double *H_B_BN_wss,
		double *alpha_wss, double *G_BN, double *grad) {
	//	H = Q(B, B);
	//	H_BN = Q(B, N);

	chrono::system_clock::time_point alfpgm_start_time =
			chrono::system_clock::now();
	double L, LQ;
	double lambda;
	double g_alpha;
	double rec;
	int fgIter = 0;
	int k;
	int nData = alfpgm_prob->nData;
	int nWss = alfpgm_param.nWss;
	double nu;
	int warnflag;

	double *alpha_wss_fpgm = (double*) (calloc(nWss, sizeof(double)));
	double *alpha_wss_prev = (double*) (calloc(nWss, sizeof(double)));
	double *alpha_H = (double*) (calloc(nWss, sizeof(double)));

	double *HBB = (double*) (calloc(nWss * nWss, sizeof(double)));
	double *grad_fg = (double*) (calloc(nWss, sizeof(double)));
	double L_inverse = 0;
	int wssPairsDoubleByte = nWss * sizeof(double);

	for (k = 0; k < nWss; k++) {
		memcpy(&HBB[k * nWss], &H_B_BN_wss[k * nData], wssPairsDoubleByte);
	}

	LQ = computeL(HBB);

	memcpy(alpha_wss_fpgm, alpha_wss, wssPairsDoubleByte);

	int OuterIter = 0;
	lambda = 0;

	nu = svm_param.C;
	rec = nu;

	double obj_val = DBL_MAX;

	bool do_outer_loop = true;

	while (do_outer_loop) {
		L = LQ + alfpgm_param.mu * nWss;
		L_inverse = 1 / L;
		memcpy(alpha_wss_prev, alpha_wss_fpgm, wssPairsDoubleByte);

		fgIter = fast_gradient(alfpgm_prob, alpha_H, HBB, G_BN, y_wss,
				alpha_wss_fpgm, grad_fg, L_inverse, lambda);

		alfpgm_param.total_inner_iter += fgIter;

		g_alpha = dotProduct(alpha_wss_fpgm, y_wss, nWss) - alfpgm_prob->beq;
		nu = calcGradNu(alpha_H, HBB, G_BN, grad, y_wss, alpha_wss_fpgm, lambda,
				g_alpha);

		lambda = lambda - alfpgm_param.mu * g_alpha;

		memcpy(alpha_wss, alpha_wss_fpgm, wssPairsDoubleByte);
		if (alfpgm_param.verbose == 2) {

			logresults(LogfileID, "a: FPGM iteration number: %d\n", OuterIter);
			logresults(LogfileID, "b: Grad Lagrangian: %15.10g\n",
					maxabsvec(grad));
			logresults(LogfileID,
					"c: Equality constraints infeasibility |g|: %15.10g\n",
					fabs(g_alpha));
			logresults(LogfileID,
					"d: Inequality constraints infeasibility max(0, -c(x)): %6.4g\n",
					get_infeasibility(alpha_wss));
			logresults(LogfileID,
					"e: Number of FPGM steps within this FISTA iteration.: %3d\n",
					fgIter);
			logresults(LogfileID, "f: Objective value : %9.6e \n", obj_val);
			logresults(LogfileID, "g: nu : %9.6e \t mu : %9.6e\n\n", nu,
					alfpgm_param.mu);
		}

		rec = fmin(rec, fmax(nu, fabs(g_alpha)));
		alfpgm_param.reqAccuracy = alfpgm_param.theta * rec;
		OuterIter++;

		if (alfpgm_param.reqAccuracy < alfpgm_param.alfpgm_accuracy) {
			warnflag = 0;
			do_outer_loop = false;
		}

		if (OuterIter >= alfpgm_param.maxouteriter) {
			warnflag = 1;
			do_outer_loop = false;
		}
	}

	alfpgm_param.total_outer_iter += OuterIter;
	free(grad_fg);
	free(HBB);
	free(alpha_wss_prev);
	free(alpha_wss_fpgm);
	free(alpha_H);

	if (alfpgm_param.verbose >= 1) {
		logresults(LogfileID,
				"Decomposition %d, ALFPGM Time : %f, nu : %g, mu : %g, exit_flag %d\n",
				alfpgm_param.ndecomp, get_elapsed_time(alfpgm_start_time), nu,
				alfpgm_param.mu, warnflag);
	}

	return (warnflag);
}

int fista_original(double *alpha_H, double lambda, double L_inverse,
		double *alpha_fpgm, svm_problem *alfpgm_prob, double *HBB, double *G_BN,
		double *grad_fg, double *y_wss) {
	double *alpha_fg = (double*) ((malloc(alfpgm_param.nWss * sizeof(double))));
	double *alpha_temp =
			(double*) ((malloc(alfpgm_param.nWss * sizeof(double))));
	memcpy(alpha_temp, alpha_fpgm, alfpgm_param.nWss * sizeof(double));
	memcpy(alpha_fg, alpha_fpgm, alfpgm_param.nWss * sizeof(double));
	int i;
	double beta_new;
	double alphadiff;
	double beta = 1.0;
	double betafactor;
	int fgIter = 0;
	double norm_val = DBL_MAX;
	bool doloopFG;
	doloopFG = calcGradStopInnerLoop(alpha_H, alfpgm_prob->beq, HBB, G_BN,
			grad_fg, y_wss, alpha_fg, lambda);
	if (alfpgm_param.verbose == 3) {
		logresults(LogfileID,
				"Inner Iteration:%03d |G|: %15.10e req_accuracy: %15.10e\n",
				fgIter, maxabsvec(grad_fg), alfpgm_param.reqAccuracy);
	}
	doloopFG = true;
	while (doloopFG && (fgIter < alfpgm_param.maxfpgmiter)) {
		beta_new = 0.5 * (1 + sqrt(1.0f + 4 * beta * beta));
		betafactor = fmin(1, (beta - 1) / beta_new);
		norm_val = 0;
//#pragma omp parallel default(none) shared(svm_param, alfpgm_param, alpha_fg, alpha_fpgm, alpha_temp, L_inverse, grad_fg, betafactor) reduction(+ : norm_val)
		{
//#pragma omp for private(i, alphadiff) schedule(guided)
			for (i = 0; i < alfpgm_param.nWss; i++) {
				alpha_temp[i] = alpha_fg[i];
				alpha_fg[i] = alpha_fpgm[i] - L_inverse * grad_fg[i];
				alpha_fg[i] = projection_operator(alpha_fg[i]);
				alphadiff = alpha_fg[i] - alpha_temp[i];
				alpha_fpgm[i] = alpha_fg[i] + betafactor * alphadiff;

				norm_val += alphadiff * alphadiff;
			}
		}

		norm_val = sqrt(norm_val) / alfpgm_param.nWss;
		beta = beta_new;
		doloopFG = norm_val > alfpgm_param.alpha_norm_diff_fpgm;
		if (doloopFG) {
			doloopFG = calcGradStopInnerLoop(alpha_H, alfpgm_prob->beq, HBB,
					G_BN, grad_fg, y_wss, alpha_fpgm, lambda);
		}
		fgIter++;
		if (alfpgm_param.verbose == 3) {
			logresults(LogfileID,
					"Inner Iteration:%03d |G|: %15.10e req_accuracy: %15.10e norm_val : %15.10e \n",
					fgIter, maxabsvec(grad_fg), alfpgm_param.reqAccuracy,
					norm_val);
		}
	}

	free(alpha_temp);
	free(alpha_fg);
	return fgIter;
}

int fista_restart(double *alpha_H, double lambda, double L_inverse,
		double *alpha_fpgm, svm_problem *alfpgm_prob, double *HBB, double *G_BN,
		double *grad_fg, double *y_wss) {
	double *alpha_fg = (double*) ((malloc(alfpgm_param.nWss * sizeof(double))));
	double *alpha_temp =
			(double*) ((malloc(alfpgm_param.nWss * sizeof(double))));
	double *alpha_fpgm_old = (double*) ((malloc(
			alfpgm_param.nWss * sizeof(double))));
	memcpy(alpha_temp, alpha_fpgm, alfpgm_param.nWss * sizeof(double));
	memcpy(alpha_fg, alpha_fpgm, alfpgm_param.nWss * sizeof(double));
	int i;
	double alphadiff;
	double beta_new;
	double beta = 1.0;
	double betafactor;
	int fgIter = 0;
	double norm_val = DBL_MAX;
	bool doloopFG;

	double v;
	doloopFG = calcGradStopInnerLoop(alpha_H, alfpgm_prob->beq, HBB, G_BN,
			grad_fg, y_wss, alpha_fg, lambda);
	if (alfpgm_param.verbose == 3) {
		logresults(LogfileID,
				"Inner Iteration:%03d |G|: %15.10e req_accuracy: %15.10e\n",
				fgIter, maxabsvec(grad_fg), alfpgm_param.reqAccuracy);
	}
	doloopFG = true;
	while (doloopFG && (fgIter < alfpgm_param.maxfpgmiter)) {
		beta_new = 0.5 * (1 + sqrt(1.0f + 4 * beta * beta));
		betafactor = fmin(1, (beta - 1) / beta_new);
		norm_val = 0;

		memcpy(alpha_fpgm_old, alpha_fpgm, alfpgm_param.nWss * sizeof(double));
//#pragma omp parallel default(none) shared(svm_param, alfpgm_param, alpha_fg, alpha_fpgm, alpha_temp, L_inverse, grad_fg, betafactor) reduction(+ : norm_val)
		{
//#pragma omp for private(i, alphadiff) schedule(guided)
			for (i = 0; i < alfpgm_param.nWss; i++) {
				alpha_temp[i] = alpha_fg[i];
				alpha_fg[i] = alpha_fpgm[i] - L_inverse * grad_fg[i];
				alpha_fg[i] = projection_operator(alpha_fg[i]);
				alphadiff = alpha_fg[i] - alpha_temp[i];
				alpha_fpgm[i] = alpha_fg[i] + betafactor * alphadiff;

				norm_val += alphadiff * alphadiff;
			}
		}

		norm_val = sqrt(norm_val) / alfpgm_param.nWss;
		beta = beta_new;

		doloopFG = norm_val > alfpgm_param.alpha_norm_diff_fpgm;
		if (doloopFG) {

			doloopFG = calcGradStopInnerLoop(alpha_H, alfpgm_prob->beq, HBB,
					G_BN, grad_fg, y_wss, alpha_fpgm, lambda);

			if (doloopFG) {
				v = 0;
				for (i = 0; i < alfpgm_param.nWss; i++) {
					v += (alpha_fpgm_old[i] - alpha_fg[i])
							* (alpha_fg[i] - alpha_temp[i]);
				}

				if (v >= 0) {
					beta = 1;
					memcpy(alpha_fpgm, alpha_fg,
							alfpgm_param.nWss * sizeof(double));
				}
			}
		}

		fgIter++;
		if (alfpgm_param.verbose == 3) {
			logresults(LogfileID,
					"Inner Iteration:%03d |G|: %15.10e req_accuracy: %15.10e\n",
					fgIter, maxabsvec(grad_fg), alfpgm_param.reqAccuracy);
		}
	}

	free(alpha_fpgm_old);
	free(alpha_temp);
	free(alpha_fg);
	return fgIter;
}

int fista_mfista(double *alpha_H, double lambda, double L_inverse,
		double *alpha_fpgm, svm_problem *alfpgm_prob, double *HBB, double *G_BN,
		double *grad_fg, double *y_wss) {
	double *alpha_fg = (double*) ((malloc(alfpgm_param.nWss * sizeof(double))));
	double *alpha_zg = (double*) ((malloc(alfpgm_param.nWss * sizeof(double))));
	double *alpha_temp =
			(double*) ((malloc(alfpgm_param.nWss * sizeof(double))));

	memcpy(alpha_fg, alpha_fpgm, alfpgm_param.nWss * sizeof(double));
	int i;
	double alphadiff;
	double beta_new;
	double new_cost;
	double beta = 1.0;
	double betafactor1, betafactor2;
	int fgIter = 0;
	double norm_val = DBL_MAX;
	bool doloopFG;
	int nWss = alfpgm_param.nWss;
	doloopFG = calcGradStopInnerLoop(alpha_H, alfpgm_prob->beq, HBB, G_BN,
			grad_fg, y_wss, alpha_fg, lambda);
	if (alfpgm_param.verbose == 3) {
		logresults(LogfileID,
				"Inner Iteration:%03d |G|: %15.10e req_accuracy: %15.10e\n",
				fgIter, maxabsvec(grad_fg), alfpgm_param.reqAccuracy);
	}
	doloopFG = true;

	double cost = objval_subprob(alpha_H, alfpgm_prob->beq, HBB, y_wss,
			alpha_fg, lambda);

	while (doloopFG && (fgIter < alfpgm_param.maxfpgmiter)) {
		beta_new = 0.5 * (1 + sqrt(1 + 4 * beta * beta));
		betafactor1 = beta / beta_new;
		betafactor2 = (beta - 1) / beta_new;
		norm_val = 0;

		memcpy(alpha_temp, alpha_fg, nWss * sizeof(double));
//#pragma omp parallel default(none) shared(nWss, alpha_fpgm, alpha_zg, L_inverse, grad_fg)
		{
//#pragma omp for private(i) schedule(guided)
			for (i = 0; i < nWss; i++) {
				alpha_zg[i] = alpha_fpgm[i] - L_inverse * grad_fg[i];
				alpha_zg[i] = projection_operator(alpha_zg[i]);
			}
		}

		new_cost = objval_subprob(alpha_H, alfpgm_prob->beq, HBB, y_wss,
				alpha_zg, lambda);
		if (cost > new_cost) {
			cost = new_cost;
			memcpy(alpha_fg, alpha_zg, nWss * sizeof(double));
		}

//#pragma omp parallel default(none) shared(nWss, alpha_temp, alpha_fpgm, alpha_zg, alpha_fg, betafactor1, betafactor2) reduction(+: norm_val)
		{
//#pragma omp for private(i, alphadiff) schedule(guided)
			for (i = 0; i < nWss; i++) {
				alphadiff = alpha_fg[i] - alpha_temp[i];
				alpha_fpgm[i] = alpha_fg[i]
						+ betafactor1 * (alpha_zg[i] - alpha_fg[i])
						+ betafactor2 * alphadiff;
				norm_val += alphadiff * alphadiff;
			}
		}

		norm_val = sqrt(norm_val) / nWss;
		beta = beta_new;
		fgIter++;
		doloopFG = norm_val > alfpgm_param.alpha_norm_diff_fpgm;
		if (doloopFG) {
			doloopFG = calcGradStopInnerLoop(alpha_H, alfpgm_prob->beq, HBB,
					G_BN, grad_fg, y_wss, alpha_fpgm, lambda);
		}
		if (alfpgm_param.verbose == 3) {
			logresults(LogfileID,
					"Inner Iteration:%03d |G|: %15.10e req_accuracy: %15.10e\n",
					fgIter, maxabsvec(grad_fg), alfpgm_param.reqAccuracy);
		}
	}

	free(alpha_zg);
	free(alpha_temp);
	free(alpha_fg);
	return fgIter;
}

int fista_rada(double *alpha_H, double lambda, double L_inverse,
		double *alpha_fpgm, svm_problem *alfpgm_prob, double *HBB, double *G_BN,
		double *grad_fg, double *y_wss) {

	bool first_hit_flag = false;
	double *alpha_fg = (double*) ((malloc(alfpgm_param.nWss * sizeof(double))));
	double *alpha_temp =
			(double*) ((malloc(alfpgm_param.nWss * sizeof(double))));
	double *alpha_fpgm_old = (double*) ((malloc(
			alfpgm_param.nWss * sizeof(double))));
	memcpy(alpha_fg, alpha_fpgm, alfpgm_param.nWss * sizeof(double));
	int i, cnt = 0;
	double alphadiff;
	double beta_new;
	double beta = 1.0;
	double betafactor;
	int fgIter = 0;
	double norm_val = DBL_MAX;
	bool doloopFG;

	double p = alfpgm_param.fista_parameters_p;
	double q = alfpgm_param.fista_parameters_q;
	double r = alfpgm_param.fista_parameters_r;
	double v;
	double xi;
	doloopFG = calcGradStopInnerLoop(alpha_H, alfpgm_prob->beq, HBB, G_BN,
			grad_fg, y_wss, alpha_fg, lambda);
	if (alfpgm_param.verbose == 3) {
		logresults(LogfileID,
				"Inner Iteration:%03d |G|: %15.10e req_accuracy: %15.10e\n",
				fgIter, maxabsvec(grad_fg), alfpgm_param.reqAccuracy);
	}
	doloopFG = true;
	while (doloopFG && (fgIter < alfpgm_param.maxfpgmiter)) {
		beta_new = 0.5 * (p + sqrt(q + r * beta * beta));
		betafactor = fmin(1, (beta - 1) / beta_new);
		norm_val = 0;
		memcpy(alpha_fpgm_old, alpha_fpgm, alfpgm_param.nWss * sizeof(double));
		memcpy(alpha_temp, alpha_fg, alfpgm_param.nWss * sizeof(double));

//#pragma omp parallel default(none) shared(svm_param, alfpgm_param, alpha_fg, alpha_fpgm, alpha_temp, L_inverse, grad_fg, betafactor) reduction(+ : norm_val)
		{
//#pragma omp for private(i, alphadiff) schedule(guided)
			for (i = 0; i < alfpgm_param.nWss; i++) {
				alpha_fg[i] = alpha_fpgm[i] - L_inverse * grad_fg[i];
				alpha_fg[i] = projection_operator(alpha_fg[i]);

				alphadiff = alpha_fg[i] - alpha_temp[i];
				alpha_fpgm[i] = alpha_fg[i] + betafactor * alphadiff;

				norm_val += alphadiff * alphadiff;
			}
		}
		norm_val = sqrt(norm_val) / alfpgm_param.nWss;

		beta = beta_new;
		doloopFG = norm_val > alfpgm_param.alpha_norm_diff_fpgm;
		if (doloopFG) {
			doloopFG = calcGradStopInnerLoop(alpha_H, alfpgm_prob->beq, HBB,
					G_BN, grad_fg, y_wss, alpha_fpgm, lambda);

			if (doloopFG) {
				v = 0;
				for (i = 0; i < alfpgm_param.nWss; i++) {
					v += (alpha_fpgm_old[i] - alpha_fg[i])
							* (alpha_fg[i] - alpha_temp[i]);
				}

				if (v >= 0) {
					cnt = cnt + 1;

					if (cnt >= 4) {
						if (!first_hit_flag) {
							double a_half = (4 + 1 * betafactor) / 5;
							xi = pow(a_half, (1 / 30));
							first_hit_flag = true;
						}

						r = r * xi;
						if (r < 3.99) {
							double beta_lim = (2 * p
									+ sqrt(r * p * p + (4 - r) * q)) / (4 - r);
							beta = fmax(2 * beta_lim, beta);
						}
					} else {
						beta = 1;
					}

					memcpy(alpha_fpgm, alpha_fg,
							alfpgm_param.nWss * sizeof(double));
				}
			}
		}
		fgIter++;
		if (alfpgm_param.verbose == 3) {
			logresults(LogfileID,
					"Inner Iteration:%03d |G|: %15.10e req_accuracy: %15.10e\n",
					fgIter, maxabsvec(grad_fg), alfpgm_param.reqAccuracy);
		}
	}

	free(alpha_fpgm_old);
	free(alpha_temp);
	free(alpha_fg);
	return fgIter;
}

int fista_greedy(double *alpha_H, double lambda, double L_inverse,
		double *alpha_fpgm, svm_problem *alfpgm_prob, double *HBB, double *G_BN,
		double *grad_fg, double *y_wss) {
	double *alpha_fg = (double*) ((malloc(alfpgm_param.nWss * sizeof(double))));
	double *alpha_temp =
			(double*) ((malloc(alfpgm_param.nWss * sizeof(double))));
	double *alpha_fpgm_old = (double*) ((malloc(
			alfpgm_param.nWss * sizeof(double))));

	memcpy(alpha_fg, alpha_fpgm, alfpgm_param.nWss * sizeof(double));

	int i;
	double alphadiff;
	double beta;
	int k = 0;
	int fgIter = 0;
	double norm_val = DBL_MAX;
	bool doloopFG;

	double S = 1;
	double xi = 0.96;
	double gamma = 1.3 * L_inverse;
	double gamma0 = L_inverse;
	double norm_diff_fpgm0 = 0;
	bool first_hit_flag = false;
	double v;
	doloopFG = calcGradStopInnerLoop(alpha_H, alfpgm_prob->beq, HBB, G_BN,
			grad_fg, y_wss, alpha_fg, lambda);

	if (alfpgm_param.verbose == 3) {
		logresults(LogfileID,
				"Inner Iteration:%03d |G|: %15.10e req_accuracy: %15.10e\n",
				fgIter, maxabsvec(grad_fg), alfpgm_param.reqAccuracy);
	}
	doloopFG = true;
	while (doloopFG && (fgIter < alfpgm_param.maxfpgmiter)) {
		k++;
		beta = fmax(2.0 / (1.0 + k / 12.0), 1);
		norm_val = 0;
		v = 0;

		memcpy(alpha_temp, alpha_fg, alfpgm_param.nWss * sizeof(double));
		memcpy(alpha_fpgm_old, alpha_fpgm, alfpgm_param.nWss * sizeof(double));

//#pragma omp parallel default(none) shared(svm_param, alfpgm_param, alpha_fg, alpha_fpgm, alpha_fpgm_old, alpha_temp, L_inverse, grad_fg, gamma, beta) reduction(+	: norm_val) reduction(+: v)
		{
//#pragma omp for private(i, alphadiff) schedule(guided)
			for (i = 0; i < alfpgm_param.nWss; i++) {
				alpha_fg[i] = alpha_fpgm[i] - gamma * grad_fg[i];
				alpha_fg[i] = projection_operator(alpha_fg[i]);

				alphadiff = alpha_fg[i] - alpha_temp[i];
				alpha_fpgm[i] = alpha_fg[i] + beta * alphadiff;

				norm_val += alphadiff * alphadiff;
				v += (alpha_fpgm_old[i] - alpha_fg[i]) * alphadiff;
			}
		}
		norm_val = sqrt(norm_val) / alfpgm_param.nWss;

		//gradient criteria
		if (v >= 0) {
			memcpy(alpha_fpgm, alpha_fg, alfpgm_param.nWss * sizeof(double));
		}

		if (!first_hit_flag) {
			norm_diff_fpgm0 = norm_val;
			first_hit_flag = true;
		}

		// safeguard
		if (norm_val > S * norm_diff_fpgm0) {
			gamma = fmax(gamma0, gamma * xi);
		}
		fgIter++;
		doloopFG = norm_val > alfpgm_param.alpha_norm_diff_fpgm;
		if (doloopFG) {
			doloopFG = calcGradStopInnerLoop(alpha_H, alfpgm_prob->beq, HBB,
					G_BN, grad_fg, y_wss, alpha_fpgm, lambda);
		}

		if (alfpgm_param.verbose == 3) {
			logresults(LogfileID,
					"Inner Iteration:%03d |G|: %15.10e req_accuracy: %15.10e\n",
					fgIter, maxabsvec(grad_fg), alfpgm_param.reqAccuracy);
		}
	}

	free(alpha_fpgm_old);
	free(alpha_temp);
	free(alpha_fg);
	return fgIter;
}

int fast_gradient(svm_problem *alfpgm_prob, double *alpha_H, double *HBB,
		double *G_BN, double *y_wss, double *alpha_fpgm, double *grad_fg,
		double L_inverse, double lambda) {

	int fgIter = 0;

	switch (alfpgm_param.fista_type) {
	case 1: {
		fgIter = fista_restart(alpha_H, lambda, L_inverse, alpha_fpgm,
				alfpgm_prob, HBB, G_BN, grad_fg, y_wss);
		break;
	}
	case 2: {
		fgIter = fista_mfista(alpha_H, lambda, L_inverse, alpha_fpgm,
				alfpgm_prob, HBB, G_BN, grad_fg, y_wss);
		break;
	}
	case 3: {
		fgIter = fista_rada(alpha_H, lambda, L_inverse, alpha_fpgm, alfpgm_prob,
				HBB, G_BN, grad_fg, y_wss);
		break;
	}
	case 4: {
		fgIter = fista_greedy(alpha_H, lambda, L_inverse, alpha_fpgm,
				alfpgm_prob, HBB, G_BN, grad_fg, y_wss);
		break;
	}
	default: {
		fgIter = fista_original(alpha_H, lambda, L_inverse, alpha_fpgm,
				alfpgm_prob, HBB, G_BN, grad_fg, y_wss);
		break;
	}
	}
	return (fgIter);
}

double calcGradNu(double *alpha_H, double *HBB, double *G_BN, double *grad,
		double *y_wss, double *alpha_vec, double lambda, double g_alpha) {

	//grad = QxAlpha + G_BN - (lambda - mu*g_alpha)*y;

	double nu = fabs(g_alpha);
	int i;

	double multiplier = (lambda - alfpgm_param.mu * g_alpha);

	cblas_dgemv(CblasRowMajor, CblasNoTrans, alfpgm_param.nWss,
			alfpgm_param.nWss, 1.0, HBB, alfpgm_param.nWss, alpha_vec, 1, 0,
			alpha_H, 1);

	for (i = 0; i < alfpgm_param.nWss; i++) {
		grad[i] = alpha_H[i] + G_BN[i] - multiplier * y_wss[i];

		if (alpha_vec[i] <= alfpgm_param.epsilon) {
			nu = fmax(nu, -grad[i]);
		} else if (alpha_vec[i] >= (svm_param.C - alfpgm_param.epsilon)) {
			nu = fmax(nu, grad[i]);
		} else {
			nu = fmax(nu, fabs(grad[i]));
		}
	}

	return (nu);
}

bool calcGradStopInnerLoop(double *alpha_H, double beq, double *HBB,
		double *G_BN, double *gradfg, double *y_wss, double *alpha_fpgm,
		double lambda) {

	double g_alpha = dotProduct(alpha_fpgm, y_wss, alfpgm_param.nWss) - beq;
	double nu = fabs(g_alpha);
	int i;

	double multiplier = (lambda - alfpgm_param.mu * g_alpha);

	bool doloopFG = false;

	cblas_dgemv(CblasRowMajor, CblasNoTrans, alfpgm_param.nWss,
			alfpgm_param.nWss, 1.0, HBB, alfpgm_param.nWss, alpha_fpgm, 1, 0,
			alpha_H, 1);

	for (i = 0; i < alfpgm_param.nWss; i++) {
		gradfg[i] = alpha_H[i] + G_BN[i] - multiplier * y_wss[i];

		if (!doloopFG) {
			if (fabs(alpha_fpgm[i]) < alfpgm_param.epsilon) {
				nu = fmax(nu, -1 * gradfg[i]);
			} else if (fabs(alpha_fpgm[i] - svm_param.C)
					< alfpgm_param.epsilon) {
				nu = fmax(nu, gradfg[i]);
			} else {
				nu = fmax(nu, fabs(gradfg[i]));
			}
			doloopFG = nu > alfpgm_param.reqAccuracy;
		}
	}
	return (doloopFG);
}

//Objective value
double objval_subprob(double *alpha_H, double beq, double *HBB, double *y_wss,
		double *alpha_fpgm, double lambda) {

	//obj_val = (0.5*QxAlpha_fpgm' -I')*alpha_fpgm - (lambda - 0.5*mu*g_alpha)*g_alpha;

	int i;
	double obj_val;
	double dsum = 0;
	double g_alpha = -beq;

	cblas_dgemv(CblasRowMajor, CblasNoTrans, alfpgm_param.nWss,
			alfpgm_param.nWss, 1.0, HBB, alfpgm_param.nWss, alpha_fpgm, 1, 0,
			alpha_H, 1);

	for (i = 0; i < alfpgm_param.nWss; i++) {
		dsum += (0.5f * alpha_H[i] - 1) * alpha_fpgm[i];

		g_alpha += y_wss[i] * alpha_fpgm[i];
	}

	obj_val = dsum - (lambda - 0.5f * alfpgm_param.mu * g_alpha) * g_alpha;
	return (obj_val);
}

double computeL(double *HBB) {
	int i, j;
	double eig_max = -DBL_MAX, ldiff = DBL_MAX, zprevious;

	double *z_eig, *x_eig;
	z_eig = (double*) (calloc(alfpgm_param.nWss, sizeof(double)));
	x_eig = (double*) (calloc(alfpgm_param.nWss, sizeof(double)));
	for (j = 0; j < alfpgm_param.nWss; j++) {
		x_eig[j] = (double) (rand()) / (double) ((RAND_MAX));
	}
	while (ldiff > .01) {
		zprevious = eig_max;

		cblas_dgemv(CblasRowMajor, CblasNoTrans, alfpgm_param.nWss,
				alfpgm_param.nWss, 1.0, HBB, alfpgm_param.nWss, x_eig, 1, 0,
				z_eig, 1);

		eig_max = z_eig[0];
		for (i = 1; i < alfpgm_param.nWss; i++) {
			eig_max = fmax(eig_max, fabs(z_eig[i]));
		}

		for (i = 0; i < alfpgm_param.nWss; i++) {
			x_eig[i] = z_eig[i] / eig_max;
		}
		ldiff = fabs(eig_max - zprevious);
	}
	free(z_eig);
	free(x_eig);
	return (eig_max);
}

double projection_operator(double alpha_fg) {
	if (alpha_fg < alfpgm_param.epsilon) {
		alpha_fg = 0;
	}
	if (alpha_fg > (svm_param.C - alfpgm_param.epsilon)) {
		alpha_fg = svm_param.C;
	}
	return alpha_fg;
}

double maxabsvec(double *Vec) {
	double max_abs_val = 0;
	int i;
	for (i = 0; i < alfpgm_param.nWss; i++) {
		max_abs_val = fmax(max_abs_val, fabs(Vec[i]));
	}
	return (max_abs_val);
}

double get_infeasibility(double *vector) {
	double Infeas = 0;
	for (int i = 0; i < alfpgm_param.nWss; i++) {
		Infeas = fmax(Infeas, fmax(-vector[i], vector[i] - svm_param.C));
	}
	return (Infeas);
}

double dotProduct(double *vecA, double *vecB, int n) {
	double dotP = 0;
	for (int k = 0; k < n; k++) {
		dotP += vecA[k] * vecB[k];
	}
	return (dotP);
}
