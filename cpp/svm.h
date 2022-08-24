#include <cmath>
#include <vector>
#include <iostream>

#include "include/svm_01_support.h"
#include "include/svm_02_SV.h"
#include "include/svm_03_nSV.h"
#include "include/svm_04_svCoef.h"
#include "include/svm_05_intercept.h"
#include "include/svm_06_svmType.h"
#include "include/svm_07_kernel.h"
#include "include/svm_08_degree.h"
#include "include/svm_09_gamma.h"
#include "include/svm_10_coef0.h"

void getSvmPredict(float* feature, int feature_len, int &dec_values);