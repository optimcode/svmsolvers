#ifndef _SVMFPGM_H_
#define _SVMFPGM_H_

#include "common.h"

using namespace std;

size_t cache_capacity;

struct ind_val {
	int index;
	double value;
};

struct head_t {
	head_t *prev, *next; // a circular list
	int len;
	double *data;
};

struct dataset {
	int nData = 0;
	int *label = NULL;
	double *X;
};

struct decision_function {
	double *alpha = NULL;
	double rho;
	int exit_flag;
};

template<typename T>
int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

FILE *LogfileID = fopen("svmfpgm.log", "w");
svm_parameter svm_param;
alfpgm_parameter alfpgm_param;

size_t cache_size;
head_t *head;
head_t lru_head;

decision_function SVMSOLVER(svm_problem *alfpgm_prob);
void updateF(double *minusF, svm_problem *alfpgm_prob, const double *H_B_BN,
		const double *alpha_differences_vector);
void find_min_count(svm_problem *alfpgm_prob);
double rbf_dist_vector(const double *Xi, const double *Xj);
int svm_predict(const double *Xi, const svm_model model, double *XDot);
double PredictError(dataset test_problem, svm_model model);
void assign_Itmp_Jtmp(svm_problem *alfpgm_prob, double *alpha, double *y_wss,
		double *alpha_wss, double *H_B_BN_temp, int nB, int Itmp, int Jtmp, double *minusF);
int get_second_order_mvp_for_i1(double *Xi, int i_up,
		vector<pair<double, int>> &Ilow, vector<int> &wss_list,
		double *minusF);
int get_second_order_mvp_for_i2(svm_problem *alfpgm_prob, double *Xi, int i_up,
		vector<pair<double, int>> &Ilow, vector<int> &wss_list,
		double *minusF);
void get_wss2_Iup_Ilow(svm_problem *alfpgm_prob, double *alpha,
		vector<int> active_list,
		vector<pair<double, int>> &Iup, double *minusF,
		vector<pair<double, int>> &Ilow);
void wss_type_0(svm_problem *alfpgm_prob, double *alpha, double *y_s,
		double *alpha_s, double *minusF, double *H_B_BN, double *G_BN);
void wss_type_1(svm_problem *alfpgm_prob, double *alpha, double *y_s,
		double *alpha_s, double *minusF, double *H_B_BN, double *G_BN);
void wss_type_2(svm_problem *alfpgm_prob, double *alpha, double *y_wss,
		double *alpha_wss, double *minusF, double *H_B_BN, double *G_BN);
double compute_final_objective_bias(svm_problem *alfpgm_prob, double *alpha);
void compose_F();
void compose_K_F(svm_problem *alfpgm_prob);

void find_Iup_Ilow(svm_problem *alfpgm_prob, vector<pair<double, int>> &Iup,
		vector<pair<double, int>> &Ilow, double *alpha, double *minusF);
int fast_get_mvp(int Itmp, vector<pair<double, int>> Ilow, double minusFi,
		vector<int> wss_list);
int sorting_ind_val_asc(const void *a, const void *b);
int sorting_ind_val_des(const void *a, const void *b);
bool calcGradStopInnerLoop(svm_problem *alfpgm_prob, double *HBN, double *G_BN,
		double *gradfg, double *y_s, double *alpha_fpgm, double lambda);
double getK_ij(svm_problem *alfpgm_prob, double *Xi, int j);
void getK_i(svm_problem *alfpgm_prob, double *H_i, int i);
void getQ_i(svm_problem *alfpgm_prob, double *H_i, int i);
void readin_Data(const char *file_name, dataset *load_data, bool file_gamma,
		bool isTraining);

int ALFPGM(svm_problem *alfpgm_prob, double *y_wss,
		double *H_B_BN_wss, double *alpha_wss, double *G_BN, double *grad);
bool is_lower_bound_alpha(double alpha_i);
bool is_upper_bound_alpha(double alpha_i);
double get_elapsed_time(std::chrono::system_clock::time_point start_time);
char* getCmdOption(char **begin, char **end, const std::string &option);
void parse_arguments(int argc, char **argv);
int get_data(svm_problem *alfpgm_prob, const int index, double **data);
void swapint(int &a, int &b);
void lru_delete(head_t *h);
void lru_insert(head_t *h);
int get_tri_index(int i, int j);
bool cmdOptionExists(char **begin, char **end, const std::string &option);
int svm_group_classes(dataset *train_data, int **label_ret, int **start_ret,
		int **count_ret, int *perm);
bool sortbyfirstdesc(const pair<double, int> &a, const pair<double, int> &b);
bool sortbyfirstcasc(const pair<double, int> &a, const pair<double, int> &b);
void popItemPair(vector<pair<double, int>> &v, int K);
double selfdotProduct(const double *vecA, int n);
#endif
