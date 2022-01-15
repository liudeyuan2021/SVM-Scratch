##
## Created by Liu Deyuan on 2021/01/15.
##

import time
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from hog_origin import feature_extraction

# (1)读取数据
data = np.load('dataset/data_float16.npz')
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

# (2)提取数据特征
print('Begin Feature Extraction')
X_train_feature = feature_extraction(X_train)
start_time = time.time()
X_test_feature = feature_extraction(X_test)
end_time = time.time()
print("{:f}s for {:d} test feature extraction".format(end_time - start_time, y_test.shape[0]))
print()

# (3)创建SVM模型
print('Begin SVM')
clf = svm.SVC() # 默认kernel为rbf，测试代码仅支持kernel为rbf

# (4)训练SVM模型
clf.fit(X_train_feature, y_train)

# (5)保存SVM模型参数
LIBSVM_IMPL = ["c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"]
print('X_test_feature', X_test_feature.shape)
print('self.support_', clf.support_.shape, clf.support_)
print('self.support_vectors_', clf.support_vectors_.shape, clf.support_vectors_)
print('self._n_support', clf._n_support)
print('self._dual_coef_', clf._dual_coef_.shape, clf._dual_coef_)
print('self._intercept_', clf._intercept_)
print('self._probA', clf._probA)
print('self._probB', clf._probB)
print('svm_type', LIBSVM_IMPL.index(clf._impl))
print('kernel', clf.kernel)
print('self.degree', clf.degree)
print('self.coef0', clf.coef0)
print('self.gamma', clf._gamma)
print('self.cache_size', clf.cache_size)
print()

np.savez('model/params.npz', support = clf.support_, SV = clf.support_vectors_, nSV = clf._n_support, 
          sv_coef = clf._dual_coef_, intercept = clf._intercept_, svm_type = LIBSVM_IMPL.index(clf._impl), 
          kernel = clf.kernel, degree = clf.degree, gamma = clf._gamma, coef0 = clf.coef0)

print(clf.break_ties, clf.decision_function_shape, len(clf.classes_), clf._sparse, callable(clf.kernel))
print()

# (6)测试模型精度
print(' -------- 原版的SVM模型测试 ---------- ')
start_time = time.time()
result = clf.predict(X_train_feature)
end_time = time.time()
print("{:f}s for {:d} train set predict".format(end_time - start_time, y_train.shape[0]))
print("{:d} positive classes in {:d} train set".format(np.sum(y_train), y_train.shape[0]))
print("accuracy_score: {:f}".format(accuracy_score(y_train, result)))
print("accuracy_number: {:d}/{:d}".format(int(accuracy_score(y_train, result, normalize=False)), len(y_train)))
print("f1_score: {:f}".format(f1_score(y_train, result)))
print()

start_time = time.time()
result = clf.predict(X_test_feature)
end_time = time.time()
print("{:f}s for {:d} test set predict".format(end_time - start_time, y_test.shape[0]))
print("{:d} positive classes in {:d} test set".format(np.sum(y_test), y_test.shape[0]))
print("accuracy_score: {:f}".format(accuracy_score(y_test, result)))
print("accuracy_number: {:d}/{:d}".format(int(accuracy_score(y_test, result, normalize=False)), len(y_test)))
print("f1_score: {:f}".format(f1_score(y_test, result)))
print()