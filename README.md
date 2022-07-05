# svmsolvers
SVM Solvers using Lagrange multiplier methods

This repository contains two methods for solving SVM
 1) svmfpgm - Augmented Langrange Fast projected gradient method (ALFPGM) and 
 2) svmnral - Non linear rescaling augmented Lagrange method
 
 
Usage: svmfpgm -f training_set_filename
		"options:\n"
		"-e test file"
		"-a 0/1 to indicate if train fail contains gamma in third row"
		"-c cost : set the parameter C of C-SVC \n"
		"-m cachesize : set cache memory size in MB (default 100)\n"
		"-o min_check_max : set min_check_max size"
		"-u mu : mu value\n"
		"-p npairs : number of pairs"
		"-t accuracy : tolerance value\n"
		"-v verbose : verbose value\n"
		"-d theta : theta value\n"
		"-g gamma : gamma value\n"
		"-y processors : processors value\n"
		"-x fista_type : fista_type type value\n"
		"-z use_cache_type : use_cache_type (0,1)\n"
		"-w wss type : working set selection type (0 or 1)\n");
    
   
Usage: svmnral -f training_set_filename
		"options:\n"
		"-e test file"
		"-a 0/1 to indicate if train fail contains gamma in third row"
		"-c cost : set the parameter C of C-SVC \n"
		"-m cachesize : set cache memory size in MB (default 100)\n"
		"-o min_check_max : set min_check_max size"
		"-u mu : mu value\n"
		"-p npairs : number of pairs"
		"-t accuracy : tolerance value\n"
		"-v verbose : verbose value\n"
		"-y processors : processors value\n"
		"-w wss type : working set selection type (0 or 1)\n");
