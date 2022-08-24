##
## Created by Liu Deyuan on 2021/01/15.
##

import time
import cv2
import numpy as np
from hog_ldy import feature_extraction
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
        self.rho = np.copy(rho) # 由于后续的代码会修改rho的值，需要使用copy避免对后续的测试造成影响
                                # 我们使用的模型nr_class=2，rho其实只有1个值，在C++中可以使用值传递

        m = nr_class * (nr_class - 1) // 2
        
        self.l = self.support.shape[0] # 支撑向量的数量
        self.SV = dense_to_libsvm(self.SV) # 支撑向量转化，和源代码对应，方便之后修改为C++代码
        self.label = np.arange(nr_class) # 类标签

        # sklearn的此处C++代码不太理解，实测中不执行此步骤也无影响
        # for i in range(self.nr_class-1):
        #     self.sv_coef[i] = self.sv_coef[0] + i * self.l

        for i in range(m):
            self.rho[i] = -self.rho[i]


def dense_to_libsvm(SV):
        nodes = []
        for i in range(SV.shape[0]):
            nodes.append(svm_node(SV.shape[1], SV[i]))
        return np.array(nodes)


def predict(X, support, SV, nSV, sv_coef, intercept, 
            svm_type=0, kernel='rbf', degree=3,
            gamma=0.1, coef0=0.0,):
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

    dec_values = copy_predict(X, model)

    return dec_values


def copy_predict(X, model):

    dec_values = np.empty(X.shape[0]) # int类型
    predict_nodes = dense_to_libsvm(X)

    for i in range(X.shape[0]):
        dec_values[i] = svm_predict(model, predict_nodes[i])

    return dec_values


def svm_predict(model, node):
    
    nr_class = model.nr_class
    l = model.l
    kvalue = np.empty(l, dtype=np.float16)
    
    for i in range(l):
        kvalue[i] = k_function(node, model.SV[i], model.param)

    start = np.empty(nr_class) # int类型
    start[0] = 0
    for i in range(1, nr_class):
        start[i] = start[i-1] + model.nSV[i-1]

    vote = np.zeros(nr_class) # int类型

    # 此处的循环是为了兼容不同的nr_class的情况
    # 我们使用的模型nr_class=2，所以实际上只会有i=0，j=1这种情况，在C++实现时可以简化掉外层的循环
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


def k_function(x, y, param):
    
    sum = 0
    dim = min(x.dim, y.dim)
    m_array = np.empty(dim, dtype=np.float16)

    for i in range(dim):
        m_array[i] = x.values[i] - y.values[i]
    sum = np.dot(m_array, m_array)

    for i in range(dim, x.dim):
        sum += x.values[i] * x.values[i]
    for i in range(dim, y.dim):
        sum += y.values[i] * y.values[i]
    
    return np.exp(-param.gamma * sum)


if __name__ == "__main__":

    # c = cv2.imread('1.png')
    # d = cv2.imread('2.png')
    # print(cv2.PSNR(c, d))

    from svm_origin import load_data, analysis_result

    # (1)读取数据
    X_train, X_test, y_train, y_test = load_data()

    import cv2
    from dataloader import read_bin, get_bin_file_with_width_and_height

    images = []
    labels = []

    # a = '/Users/liudeyuan/Desktop/商汤杂项/SVM/input/rsz_warped_400x300.bin'
    # width, height = map(int, a.split('.')[0].split('_')[-1].split('x'))
    # image = read_bin(a, width, height)

    a = '/Users/liudeyuan/Desktop/商汤杂项/SVM/input/rsz_warped.png'
    image = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
    image = np.array(image, dtype=np.float32)
    image = image / 255.0

    image = np.array(image, np.float32)
    # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    b = np.array(image * 255, dtype=np.uint8)
    cv2.imwrite('1.png', b)

    image = cv2.resize(image, dsize=(400, 300), interpolation=cv2.INTER_LINEAR)
    image = np.array(image, np.float16)

    images.append(image)
    labels.append(1)
    X_test = np.array(images)
    y_test = np.array(labels)

    X_train = X_train[-1:]
    y_train = y_train[-1:]
    # X_test = X_test[:2]
    # y_test = y_test[:2]

    # print(y_train, y_test)

    # (2)提取数据特征
    # print('Begin Feature Extraction')
    # start_time = time.time()
    # X_train_feature = feature_extraction(X_train)
    # end_time = time.time()
    # print(f'{end_time - start_time:.6f}s for {y_train.shape[0]} samples feature extraction')
    # print()
    
    start_time = time.time()
    X_test_feature = feature_extraction(X_test)
    end_time = time.time()
    # print(f'{end_time - start_time:.6f}s for {y_test.shape[0]} samples feature extraction')
    # print()

    # print('------------')
    # print('Hog Feature:')
    # for i in range(8):
    #     for j in range(8):
    #         print(X_test_feature[0][i * 8 + j], end=' ')
    #     print()
    # print('------------')
    # print()

    # (3)加载SVM模型参数
    params = np.load('model/params.npz')
    support, SV, nSV, sv_coef, intercept, \
    svm_type, kernel, degree, gamma, coef0 = \
    params['support'], params['SV'], params['nSV'], \
    params['sv_coef'], params['intercept'], \
    params['svm_type'], params['kernel'], params['degree'], \
    params['gamma'], params['coef0']
    
    # (4)测试模型精度
    # print(' -------- 自行实现的SVM模型测试(float16) ---------- ')
    # start_time = time.time()
    # result = predict(X = X_train_feature, 
    #                  support = support,
    #                  SV = SV,
    #                  nSV = nSV, 
    #                  sv_coef = sv_coef,
    #                  intercept = intercept,  
    #                  svm_type = svm_type, 
    #                  kernel = kernel, 
    #                  degree = degree, 
    #                  gamma = gamma, 
    #                  coef0 = coef0)
    # end_time = time.time()
    # analysis_result(y_train, result, end_time-start_time)
    
    start_time = time.time()
    result = predict(X = X_test_feature, 
                     support = support,
                     SV = SV,
                     nSV = nSV, 
                     sv_coef = sv_coef,
                     intercept = intercept,  
                     svm_type = svm_type, 
                     kernel = kernel, 
                     degree = degree, 
                     gamma = gamma, 
                     coef0 = coef0)
    end_time = time.time()
    analysis_result(y_test, result, end_time-start_time)