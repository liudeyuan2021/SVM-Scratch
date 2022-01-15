from cgi import parse_multipart
from math import degrees
import numpy as np

def predict(X, support, SV, nSV, sv_coef, intercept, 
            probA=np.empty(0), probB=np.empty(0), 
            svm_type=0, kernel='rbf', degree=3,
            gamma=0.1, coef0=0.0,
            class_weight=np.empty(0), sample_weight=np.empty(0), cache_size=100.):
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
    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default="rbf"
        Kernel to use in the model: linear, polynomial, RBF, sigmoid
        or precomputed.
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
    # svm_parameter param
    # svm_model *model
    # int rv

    # np.ndarray[np.int32_t, ndim=1, mode='c'] class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)

    # set_predict_params(&param, svm_type, kernel, degree, gamma, coef0,
    #                    cache_size, 0, <int>class_weight.shape[0],
    #                    class_weight_label.data, class_weight.data)
    # model = set_model(&param, <int> nSV.shape[0], SV.data, SV.shape,
    #                   support.data, support.shape, sv_coef.strides,
    #                   sv_coef.data, intercept.data, nSV.data, probA.data, probB.data)
    # BlasFunctions blas_functions
    # blas_functions.dot = _dot[double]
    # #TODO: use check_model
    try:
        dec_values = np.empty(X.shape[0])
    #     with nogil:
    #         rv = copy_predict(X.data, model, X.shape, dec_values.data, &blas_functions)
    #     if rv < 0:
    #         raise MemoryError("We've run out of memory")
    finally:
        pass
    #     free_model(model)

    return dec_values

if __name__ == "__main__":
    params = np.load('model/params.npz')
    X_test_feature, support, SV, nSV, sv_coef, intercept, probA, probB, \
    svm_type, kernel, degree, gamma, coef0, cache_size = \
    params['X_test_feature'], params['support'], params['SV'], params['nSV'], \
    params['sv_coef'], params['intercept'], params['probA'], params['probB'], \
    params['svm_type'], params['kernel'], params['degree'], params['gamma'], \
    params['coef0'], params['cache_size']
    

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
    print(result)