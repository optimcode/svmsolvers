#ifndef _FISTA_H_
#define _FISTA_H_

#include "common.h"

using namespace std;

extern FILE *LogfileID;
extern svm_parameter svm_param;
extern alfpgm_parameter alfpgm_param;

double get_infeasibility(double *alpha);
double maxabsvec(double *Vec);
double projection_operator(double alpha_fg);
double computeL(double *HBB);
bool calcGradStopInnerLoop(double *alpha_H, double beq, double *HBB, double *G_BN,
		double *gradfg, double *y_wss, double *alpha_fpgm, double lambda);
double calcGradNu(double *alpha_H, double *HBB, double *G_BN,
						double *grad, double *y_s, double *alpha_vec, double lambda, double alpha_y);
int fast_gradient(svm_problem *alfpgm_prob, double *alpha_H, double *HBB, double *G_BN,
				  double *y_s, double *alpha_fpgm, double *grad_fg, double L_inverse,
				  double lambda);
int fista_original(double *alpha_H, double lambda, double L_inverse, double *alpha_fpgm,
		svm_problem *alfpgm_prob, double *HBB, double *G_BN, double *grad_fg,
		double *y_wss);
int fista_restart(double *alpha_H, double lambda, double L_inverse, double *alpha_fpgm,
		svm_problem *alfpgm_prob, double *HBB, double *G_BN, double *grad_fg,
		double *y_wss);
int fista_mfista(double *alpha_H, double lambda, double L_inverse, double *alpha_fpgm,
		svm_problem *alfpgm_prob, double *HBB, double *G_BN, double *grad_fg,
		double *y_wss);
int fista_rada(double *alpha_H, double lambda, double L_inverse, double *alpha_fpgm,
		svm_problem *alfpgm_prob, double *HBB, double *G_BN, double *grad_fg,
		double *y_wss);
int fista_greedy(double *alpha_H, double lambda, double L_inverse, double *alpha_fpgm,
		svm_problem *alfpgm_prob, double *HBB, double *G_BN, double *grad_fg,
		double *y_wss) ;
double objval_subprob(double *alpha_H, double beq, double *HBB, double *y_wss,
		double *alpha_fpgm, double lambda);
double dotProduct(double *vecA, double *vecB, int n);
extern double get_elapsed_time(const chrono::system_clock::time_point start_time);
#endif
