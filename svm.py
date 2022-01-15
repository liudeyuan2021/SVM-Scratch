import time
import numpy as np
from scipy.linalg.blas import ddot
from utils import feature_extraction
from sklearn.metrics import accuracy_score, f1_score

class svm_node:

    def __init__(self, dim, values):
        self.dim = dim
        self.values = values


class svm_predict_parameter:

    def __init__(self, svm_type, kernel_type, degree, gamma, coef0):
        self.svm_type = svm_type
        self.kernel_type = kernel_type
        self.degree = degree    # for poly
        self.gamma = gamma      # for poly/rbf/sigmoid
        self.coef0 = coef0      # for poly/sigmoid

class svm_predict_model:

    def __init__(self, param, nr_class, support, SV, nSV, sv_coef, rho):
        self.param = param
        self.nr_class = nr_class
        self.support = support
        self.SV = SV
        self.nSV = nSV
        self.sv_coef = sv_coef
        self.rho = rho

        m = nr_class * (nr_class - 1) // 2
        
        self.l = self.support.shape[0] # 支撑向量的数量
        self.SV = dense_to_libsvm(self.SV) # 支撑向量转化，和源代码对应，方便之后修改为C++代码
        self.label = np.arange(nr_class) # 类标签

        for i in range(self.nr_class-1):
            self.sv_coef[i] = self.sv_coef[i] + i * self.l

        for i in range(m):
            self.rho[i] = -self.rho[i]


def dense_to_libsvm(SV):
        nodes = []
        for i in range(SV.shape[0]):
            nodes.append(svm_node(SV.shape[1], SV[i]))
        return np.array(nodes)


def predict(X, support, SV, nSV, sv_coef, intercept, 
            probA=np.empty(0), probB=np.empty(0), 
            svm_type=0, kernel='rbf', degree=3,
            gamma=0.1, coef0=0.0,
            class_weight=np.empty(0), sample_weight=np.empty(0), cache_size=100.0):
    """
    Predict target values of X given a model (low-level method)
    Parameters
    ----------
    X : array-like, dtype=float of shape (n_samples, n_features)
    support : array of shape (n_support,)
        Index of support vectors in training set.
    SV : array of shape (n_support, n_features)
        Support vectors.
    nSV : array of shape (n_class,)
        Number of support vectors in each class.
    sv_coef : array of shape (n_class-1, n_support)
        Coefficients of support vectors in decision function.
    intercept : array of shape (n_class*(n_class-1)/2)
        Intercept in decision function.
    probA, probB : array of shape (n_class*(n_class-1)/2,)
        Probability estimates.
    svm_type : {0, 1, 2, 3, 4}, default=0
        Type of SVM: C_SVC, NuSVC, OneClassSVM, EpsilonSVR or NuSVR
        respectively.
    kernel : {'linear', 'rbf', 'poly', 'sigmoid'}, default="rbf"
        Kernel to use in the model: linear, polynomial, RBF, sigmoid.
    degree : int32, default=3
        Degree of the polynomial kernel (only relevant if kernel is
        set to polynomial).
    gamma : float64, default=0.1
        Gamma parameter in rbf, poly and sigmoid kernels. Ignored by other
        kernels.
    coef0 : float64, default=0.0
        Independent parameter in poly/sigmoid kernel.
    Returns
    -------
    dec_values : array
        Predicted values.
    """
    param = svm_predict_parameter(svm_type, kernel, degree, gamma, coef0)
    model = svm_predict_model(param, nSV.shape[0], support, SV, nSV, sv_coef, intercept)
    blas_function = ddot # 之后还需改成自己实现的版本

    # np.ndarray[np.int32_t, ndim=1, mode='c'] class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)

    # set_predict_params(&param, svm_type, kernel, degree, gamma, coef0,
    #                    cache_size, 0, <int>class_weight.shape[0],
    #                    class_weight_label.data, class_weight.data)
    # model = set_model(&param, <int> nSV.shape[0], SV.data, SV.shape,
    #                   support.data, support.shape, sv_coef.strides,
    #                   sv_coef.data, intercept.data, nSV.data, probA.data, probB.data)
    # BlasFunctions blas_functions
    # blas_functions.dot = _dot[double]

    dec_values = copy_predict(X, model, blas_function)
    # copy_predict(X.data, model, X.shape, dec_values.data, &blas_functions)
    #     with nogil:
    #         rv = copy_predict(X.data, model, X.shape, dec_values.data, &blas_functions)
    #     if rv < 0:
    #         raise MemoryError("We've run out of memory")

    return dec_values


def copy_predict(X, model, blas_function):

    dec_values = np.empty(X.shape[0])
    predict_nodes = dense_to_libsvm(X)

    for i in range(X.shape[0]):
        dec_values[i] = svm_predict(model, predict_nodes[i], blas_function)

    return dec_values


def svm_predict(model, node, blas_function):
    
    nr_class = model.nr_class
    l = model.l
    kvalue = np.empty(l)
    
    for i in range(l):
        kvalue[i] = k_function(node, model.SV[i], model.param, blas_function)
    
    start = np.empty(nr_class)
    start[0] = 0
    for i in range(1, nr_class):
        start[i] = start[i-1] + model.nSV[i-1]

    vote = np.zeros(nr_class)

    p = 0
    for i in range(nr_class):
        for j in range(i+1, nr_class):
            sum = 0
            si = start[i]
            sj = start[j]
            ci = model.nSV[i]
            cj = model.nSV[j]

            coef1 = model.sv_coef[j-1]
            coef2 = model.sv_coef[i]

            for k in range(ci):
                sum += coef1[int(si + k)] * kvalue[int(si + k)]
            for k in range(cj):
                sum += coef2[int(sj + k)] * kvalue[int(sj + k)]  

            sum -= model.rho[p]
            
            if sum > 0:
                vote[i] += 1
            else:
                vote[j] += 1
            
            p += 1

    vote_max_id = 0
    for i in range(1, nr_class):
        if vote[i] > vote[vote_max_id]:
            vote_max_id = i
    
    return model.label[vote_max_id]

def k_function(x, y, param, blas_function):
    
    sum = 0
    dim = min(x.dim, y.dim)
    m_array = np.empty(dim)

    for i in range(dim):
        m_array[i] = x.values[i] * y.values[i]
    sum = blas_function(dim, m_array, 1, m_array, 1)

    for i in range(dim, x.dim):
        sum += x.values[i] * x.values[i]
    for i in range(dim, y.dim):
        sum += y.values[i] * y.values[i]
    
    return np.exp(-param.gamma * sum)

if __name__ == "__main__":
    # (1)读取数据
    data = np.load('dataset/data_float16.npz')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    # (2)提取数据特征
    print('Begin Feature Extraction')
    X_train_feature = feature_extraction(X_train, feature='HOG')
    start_time = time.time()
    X_test_feature = feature_extraction(X_test, feature='HOG')
    end_time = time.time()
    print("{:f}s for {:d} test feature extraction".format(end_time - start_time, y_test.shape[0]))
    print()

    X_train_feature = np.array(X_train_feature)
    X_test_feature = np.array(X_test_feature)

    # (3)测试模型精度
    params = np.load('model/params.npz')
    X_test_feature, support, SV, nSV, sv_coef, intercept, probA, probB, \
    svm_type, kernel, degree, gamma, coef0, cache_size = \
    params['X_test_feature'], params['support'], params['SV'], params['nSV'], \
    params['sv_coef'], params['intercept'], params['probA'], params['probB'], \
    params['svm_type'], params['kernel'], params['degree'], params['gamma'], \
    params['coef0'], params['cache_size']
    
    start_time = time.time()
    result = predict(X = X_test_feature, 
                     support = support,
                     SV = SV,
                     nSV = nSV, 
                     sv_coef = sv_coef,
                     intercept = intercept, 
                     probA = probA, 
                     probB = probB, 
                     svm_type = svm_type, 
                     kernel = kernel, 
                     degree = degree, 
                     gamma = gamma, 
                     coef0 = coef0, 
                     cache_size = cache_size)
    end_time = time.time()
    print("{:f}s for {:d} tests predict".format(end_time - start_time, y_test.shape[0]))
    print("{:d} positive classes in {:d} tests".format(np.sum(y_test), y_test.shape[0]))
    print("accuracy_score: {:f}".format(accuracy_score(y_test, result)))
    print("accuracy_number: {:d}/{:d}".format(int(accuracy_score(y_test, result, normalize=False)), len(y_test)))
    print("f1_score: {:f}".format(f1_score(y_test, result)))
    print()