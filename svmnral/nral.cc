#include "nral.h"

using namespace std;

int test_new_alpha_step(double lambda, bool do_get_phi_loop, int nWss,
		int wssPairsDoubleByte, double *alpha_Q, svm_problem *nral_prob,
		double *Q, double *G_BN, double *y_wss, double *alpha_wss_nral,
		double *lambda_l, double *lambda_u, double *alpha_wss_nral_new,
		double *x_sol, double mu) {
	double step = nral_param.step;
	double obj_val_test, obj_val_new;
	int exit_flag = -1;
	obj_val_test = objval_subprob(alpha_Q, nral_prob->beq, Q, G_BN, y_wss,
			alpha_wss_nral, lambda, lambda_l, lambda_u, mu);
	do_get_phi_loop = true;
	while (do_get_phi_loop) {
		for (int i = 0; i < nWss; i++) {
			alpha_wss_nral_new[i] = alpha_wss_nral[i] - step * x_sol[i];
		}
		obj_val_new = objval_subprob(alpha_Q, nral_prob->beq, Q, G_BN, y_wss,
				alpha_wss_nral_new, lambda, lambda_l, lambda_u, mu);
		if (obj_val_new > obj_val_test) {
			step = step * nral_param.stepdecr;
			if (step < nral_param.minStep) {
				do_get_phi_loop = false;
				exit_flag = 1;
			}
		} else {
			do_get_phi_loop = false;

			memcpy(alpha_wss_nral, alpha_wss_nral_new, wssPairsDoubleByte);
			exit_flag = 0;
		}
	}

	return (exit_flag);
}

int newton_loop(double lambda, double g_alpha, int nWss, int wssPairsDoubleByte,
		double *alpha_Q, double *Q, double *G_BN, double *grad, double *y_wss,
		double *alpha_wss_nral, double *lambda_l, double *lambda_u,
		svm_problem *nral_prob, double &nu, double *alpha_wss_prev_inner,
		double *H, double *x_sol, double *alpha_wss_nral_new, double mu) {

	int inner_newton_count = 0;
	double y_alpha = dotProduct(alpha_wss_nral, y_wss, nral_param.nWss);
	g_alpha = y_alpha - nral_prob->beq;
	nu = calcGradNu(alpha_Q, Q, G_BN, grad, y_wss, alpha_wss_nral, lambda,
			lambda_l, lambda_u, mu, g_alpha);

	int exit_flag = -1;
	int armijo_exit_flag = -1;
	double alpha_diff;
	double alphadiffnorm;
	bool do_newton_loop = true;
	bool do_get_phi_loop = true;

	do_newton_loop = true;

	while (do_newton_loop) {
		//primal regularization
		memcpy(alpha_wss_prev_inner, alpha_wss_nral, wssPairsDoubleByte);
		inner_newton_count++;
		compute_hessian(nWss, H, Q, y_wss, lambda_l, alpha_wss_nral, lambda_u,
				mu);
		exit_flag = solve_linear(grad, H, x_sol);
		if (exit_flag > 0) {
			exit_flag = 5;
			break;
		}

		armijo_exit_flag = test_new_alpha_step(lambda, do_get_phi_loop, nWss,
				wssPairsDoubleByte, alpha_Q, nral_prob, Q, G_BN, y_wss,
				alpha_wss_nral, lambda_l, lambda_u, alpha_wss_nral_new, x_sol,
				mu);

		if (armijo_exit_flag == 0) {
			y_alpha = dotProduct(alpha_wss_nral, y_wss, nral_param.nWss);
			g_alpha = y_alpha - nral_prob->beq;
			nu = calcGradNu(alpha_Q, Q, G_BN, grad, y_wss, alpha_wss_nral,
					lambda, lambda_l, lambda_u, mu, g_alpha);

			alphadiffnorm = 0;

			for (int i = 0; i < nWss; i++) {
				alpha_diff = alpha_wss_prev_inner[i] - alpha_wss_nral[i];
				alphadiffnorm += (alpha_diff * alpha_diff);
			}

			alphadiffnorm = sqrt(alphadiffnorm);
			if (alphadiffnorm < nral_param.alpha_norm_diff_newton) {
				do_newton_loop = false;
				exit_flag = 2;
			}

			if (do_newton_loop
					&& (nu
							< fmax(nral_param.nral_accuracy,
									nral_param.reqAccuracyFactor
											* nral_param.reqAccuracy))) {
				do_newton_loop = false;
				exit_flag = 0;
			}

			if (do_newton_loop
					&& (inner_newton_count > nral_param.maxinneriter)) {
				do_newton_loop = false;
				exit_flag = 1;
			}

		} else {
			do_newton_loop = false;
			exit_flag = 4;
		}

	}

	nral_param.total_inner_iter += inner_newton_count;

	if (nral_param.verbose == 3) {
		logresults(LogfileID, "Newton_exit_flag %d, inner_newton_count : %d\n",
				exit_flag, inner_newton_count);
	}

	return (exit_flag);

}

int NRAL(svm_problem *nral_prob, double *y_wss, double *H_B_BN_wss,
		double *alpha_wss_nral, double *G_BN, double *grad) {
	//	H = Q(B, B);
	//	H_BN = Q(B, N);

	chrono::system_clock::time_point nral_start_time =
			chrono::system_clock::now();

	int nData = nral_prob->nData;
	int nWss = nral_param.nWss;
	int warnflag = 0;
	double mu = nral_param.mu_init;

	double *alpha_wss_prev = (double*) (calloc(nWss, sizeof(double)));
	double *alpha_wss_prev_inner = (double*) (calloc(nWss, sizeof(double)));
	double *alpha_wss_nral_new = (double*) (calloc(nWss, sizeof(double)));
	int wssPairsDoubleByte = nWss * sizeof(double);

	double *lambda_l = (double*) (calloc(nWss, sizeof(double)));
	double *lambda_u = (double*) (calloc(nWss, sizeof(double)));
	double *H = (double*) (calloc(nWss * nWss, sizeof(double)));
	double *Q = (double*) (calloc(nWss * nWss, sizeof(double)));
	double *alpha_Q = (double*) (calloc(nWss, sizeof(double)));
	double *x_sol = (double*) (calloc(nWss, sizeof(double)));
	int k;
	for (k = 0; k < nWss; k++) {
		lambda_l[k] = 1.0;
		lambda_u[k] = 1.0;
		memcpy(&Q[k * nWss], &H_B_BN_wss[k * nData], wssPairsDoubleByte);
	}

	// % svm_nr_newton  Computes nonlinear resccaling with newton inner optimization.
	double lambda = 0;            //Lagrange multipliers

	int total_outer_iter = 0;   //Total outer count iteration counter

	bool do_outer_loop = true;
	double norm_diff;
	double alpha_diff;
	double y_alpha, g_alpha;
	double nu;
	int newton_exit_flag;

	y_alpha = dotProduct(alpha_wss_nral, y_wss, nral_param.nWss);
	g_alpha = y_alpha - nral_prob->beq;

	nu = calcGradNu(alpha_Q, Q, G_BN, grad, y_wss, alpha_wss_nral, lambda,
			lambda_l, lambda_u, mu, g_alpha);

	nral_param.reqAccuracy = nu;
	while (do_outer_loop) {
		total_outer_iter++;

		memcpy(alpha_wss_prev, alpha_wss_nral, wssPairsDoubleByte);

		newton_exit_flag = newton_loop(lambda, g_alpha, nWss,
				wssPairsDoubleByte, alpha_Q, Q, G_BN, grad, y_wss,
				alpha_wss_nral, lambda_l, lambda_u, nral_prob, nu,
				alpha_wss_prev_inner, H, x_sol, alpha_wss_nral_new, mu);

		if (newton_exit_flag == 5) {
			do_outer_loop = false; //solver error exit
			warnflag = 5;
		} else {
			//    update lambda
			for (int i = 0; i < nWss; i++) {
				lambda_l[i] = lambda_l[i]
						* get_gpsi_val(mu * alpha_wss_nral[i]);
				lambda_u[i] = lambda_u[i]
						* get_gpsi_val(mu * (svm_param.C - alpha_wss_nral[i]));
			}

			y_alpha = dotProduct(alpha_wss_nral, y_wss, nral_param.nWss);
			g_alpha = y_alpha - nral_prob->beq;
			nu = calcGradNu(alpha_Q, Q, G_BN, grad, y_wss, alpha_wss_nral,
					lambda, lambda_l, lambda_u, mu, g_alpha);

			nral_param.reqAccuracy = nu;

			if ((total_outer_iter > nral_param.maxouteriter) && do_outer_loop) {
				do_outer_loop = false;
				warnflag = 1;
			}
			lambda = lambda - mu * g_alpha;
			if ((nral_param.reqAccuracy < nral_param.nral_accuracy)
					&& do_outer_loop) {
				do_outer_loop = false;
				warnflag = 0;

				mu = fmin(mu * nral_param.mu_incr, nral_param.mu_max);

			} else {
				mu = fmax(mu * nral_param.mu_decr, nral_param.mu_min);

			}

			if (do_outer_loop) {
				norm_diff = 0;
				for (int i = 0; i < nWss; i++) {
					alpha_diff = alpha_wss_nral[i] - alpha_wss_prev[i];
					norm_diff += (alpha_diff * alpha_diff);
				}

				norm_diff = sqrt(norm_diff);
				if ((norm_diff < nral_param.alpha_norm_diff_nral)
						&& do_outer_loop) {
					do_outer_loop = false;
					warnflag = 2;
				}
			}

		}
	}

	if (nral_param.verbose >= 1) {
		logresults(LogfileID,
				"Decomposition %d, NRAL Time : %f, nu : %g, mu : %g\n",
				nral_param.ndecomp, get_elapsed_time(nral_start_time), nu, mu);
	}

	nral_param.total_outer_iter += total_outer_iter;

	free(alpha_wss_prev);
	free(alpha_wss_nral_new);
	free(alpha_wss_prev_inner);
	free(lambda_l);
	free(lambda_u);
	free(H);
	free(Q);
	free(alpha_Q);
	free(x_sol);
	return (warnflag);
}

double calcGradNu(double *alpha_Q, double *Q, double *G_BN, double *grad,
		double *y_wss, double *alpha_wss_nral, double lambda, double *lambda_l,
		double *lambda_u, double mu, double g_alpha) {

	//GN = (Q*alpha + G_BN) + mu*g*alpha - get_g_psi(mu*alpha).*lambda_l + get_g_psi(mu*(C -alpha)).*lambda_u - lambda*y;

	double nu = fabs(g_alpha);
	int i;
	double gnorm = 0;

	double multiplier = (lambda - mu * g_alpha);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, nral_param.nWss, nral_param.nWss,
			1.0, Q, nral_param.nWss, alpha_wss_nral, 1, 0, alpha_Q, 1);

	for (i = 0; i < nral_param.nWss; i++) {
		grad[i] = alpha_Q[i] + G_BN[i] - multiplier * y_wss[i]
				- get_gpsi_val(mu * alpha_wss_nral[i]) * lambda_l[i]
				+ get_gpsi_val(mu * (svm_param.C - alpha_wss_nral[i]))
						* lambda_u[i];
		if (alpha_wss_nral[i] <= nral_param.epsilon) {
			nu = fmax(nu, -grad[i]);
		} else if (alpha_wss_nral[i] >= (svm_param.C - nral_param.epsilon)) {
			nu = fmax(nu, grad[i]);
		} else {
			nu = fmax(nu, fabs(grad[i]));
		}
		gnorm += grad[i] * grad[i];
	}

	nu = fmax(nu, sqrt(gnorm));
	return (nu);
}

//Objective value
double objval_subprob(double *alpha_Q, double beq, double *Q, double *G_BN,
		double *y_wss, double *alpha_wss_nral, double lambda, double *lambda_l,
		double *lambda_u, double mu) {

	//  obj_val = (0.5*alpha'*Q + G_BN')*alpha + 0.5*mu*((y'*alpha)^2) - ([lambda_l; lambda_u]'*get_psi(mu*[alpha; (C-alpha)]))/mu - lambda*(y'*alpha - beq);

	int i;
	double obj_val;
	double dsum = 0;
	double negativemuninv = -1 / mu;
	double y_alpha = dotProduct(alpha_wss_nral, y_wss, nral_param.nWss);
	double g_alpha = y_alpha - beq;
	cblas_dgemv(CblasRowMajor, CblasNoTrans, nral_param.nWss, nral_param.nWss,
			1.0, Q, nral_param.nWss, alpha_wss_nral, 1, 0, alpha_Q, 1);

	for (i = 0; i < nral_param.nWss; i++) {
		dsum +=
				(0.5f * alpha_Q[i] + G_BN[i]) * alpha_wss_nral[i]
						+ negativemuninv
								* (lambda_l[i]
										* get_psi_val(mu * alpha_wss_nral[i])
										+ lambda_u[i]
												* get_psi_val(
														mu
																* (svm_param.C
																		- alpha_wss_nral[i])));
	}

	obj_val = dsum - lambda * g_alpha + 0.5 * mu * g_alpha * g_alpha;
	return (obj_val);
}

double get_psi_val(double t) {
	double psi;

	if (t >= -0.5) {
		psi = log(t + 1);
	} else {
		psi = -2 * t * t + log(0.5) + 0.5;
	}

	return (psi);
}

double get_gpsi_val(double t) {
	double gpsi;
	if (t >= -0.5) {
		gpsi = 1.0 / (t + 1);
	} else {
		gpsi = -4 * t;
	}

	return (gpsi);
}

double get_hpsi_val(double t) {
	double hpsi;
	if (t >= -0.5) {
		hpsi = -1.0 / ((t + 1) * (t + 1));
	} else {
		hpsi = -4;
	}
	return (hpsi);
}

double dotProduct(double *vecA, double *vecB, int n) {
	double dotP = cblas_ddot(n, vecA, 1, vecB, 1);
	return (dotP);
}

void compute_hessian(int nWss, double *H, double *Q, double *y_wss,
		double *lambda_l, double *alpha_wss_nral, double *lambda_u, double mu) {

	//HN = Q + mu*(Aeq_2) - mu * diag(lambda_l.*get_h_psi(mu*alpha) + lambda_u.*get_h_psi(mu*(C - alpha))));

	double kappa = nral_param.kappa;
	double C = svm_param.C;
	int i, k, j;

//#pragma omp parallel  for default(none) shared(C, Q, H, mu, y_wss, nWss) private(i, k, j) schedule(guided)
	for (i = 0; i < nWss * nWss; i++) {

		k = i / nWss;
		j = i % nWss;
		H[i] = Q[i] + mu * y_wss[j] * y_wss[k];
		if (k == j) {
			H[k * nWss + k] +=
					(kappa
							- mu
									* (lambda_l[k]
											* get_hpsi_val(
													mu * alpha_wss_nral[k])
											+ lambda_u[k]
													* get_hpsi_val(
															mu
																	* (C
																			- alpha_wss_nral[k]))));

		}
	}

}

int solve_linear(double *grad, double *H, double *sol) {

#ifdef USE_MKL
	MKL_INT *ipiv = (MKL_INT*) (calloc(nral_param.nWss, sizeof(MKL_INT)));
	MKL_INT n = nral_param.nWss, nrhs = 1, lda = nral_param.nWss, ldb =nrhs, info;
#else
	int *ipiv = (int*) (calloc(nral_param.nWss, sizeof(int)));
	int n = nral_param.nWss, nrhs = 1, lda = nral_param.nWss, ldb = nrhs, info;
#endif

	memcpy(sol, grad, n * sizeof(double));

	info = LAPACKE_dsysv( LAPACK_ROW_MAJOR, 'L', n, nrhs, H, lda, ipiv, sol,
			ldb);

	free(ipiv);

	return (info);

}
