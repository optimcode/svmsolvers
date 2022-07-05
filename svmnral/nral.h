#ifndef _NRAL_H_
#define _NRAL_H_

#include "common.h"

using namespace std;

extern FILE *LogfileID;
extern svm_parameter svm_param;
extern nral_parameter nral_param;

extern double get_elapsed_time(
		const chrono::system_clock::time_point start_time);
void compute_hessian(int nWss, double *H, double *Q, double *y_wss,
		double *lambda_l, double *alpha_wss_nral, double *lambda_u, double mu);
int solve_linear(double *grad, double *H, double *dxs);
double get_psi_val(double t);
double get_gpsi_val(double t);
double get_hpsi_val(double t);
int newton_loop(double lambda, double g_alpha, int nWss,
		int wssPairsDoubleByte, double *alpha_H,
		double *Q, double *G_BN, double *grad, double *y_wss,
		double *alpha_wss_nral, double *lambda_l, double *lambda_u,
		svm_problem *nral_prob, double &nu,
		double *alpha_wss_prev_inner, double *H,
		double *x_sol, double *alpha_wss_nral_new, double mu);

int test_new_alpha_step(double lambda, bool do_get_phi_loop,
		int nWss, int wssPairsDoubleByte, double *alpha_H,
		svm_problem *nral_prob, double *Q, double *G_BN, double *y_wss,
		double *alpha_wss_nral, double *lambda_l, double *lambda_u,
		double *alpha_wss_nral_new, double *x_sol, double mu);

double objval_subprob(double *alpha_H, double beq, double *Q, double *G_BN,
		double *y_wss, double *alpha_wss_nral, double lambda, double *lambda_l,
		double *lambda_u, double mu);
double dotProduct(double *vecA, double *vecB, int n);
extern double get_elapsed_time(
		const chrono::system_clock::time_point start_time);

double calcGradNu(double *alpha_H, double *Q, double *G_BN, double *grad,
		double *y_wss, double *alpha_wss_nral, double lambda, double *lambda_l,
		double *lambda_u, double mu, double y_alpha);
double calcGradNorm(svm_problem *nral_prob, double *grad, double *alpha_vec,
		double g_alpha);
#endif
