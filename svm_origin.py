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

data_new = np.load('dataset/data_float16_new.npz')
X_new, y_new = data_new['X_new'], data_new['y_new']

# (2)提取数据特征
print('Begin Feature Extraction')
X_train_feature = feature_extraction(X_train)

start_time = time.time()
X_test_feature = feature_extraction(X_test)
end_time = time.time()
print("{:f}s for {:d} test set feature extraction".format(end_time - start_time, y_test.shape[0]))
print()

start_time = time.time()
X_new_feature = feature_extraction(X_new)
end_time = time.time()
print("{:f}s for {:d} new set feature extraction".format(end_time - start_time, y_new.shape[0]))
print()

print(X_new_feature.shape)

# (3)创建SVM模型
print('Begin SVM')
clf = svm.SVC() # 默认kernel为rbf，测试代码仅支持kernel为rbf

# (4)训练SVM模型
clf.fit(X_train_feature, y_train)

# (5)保存SVM模型参数
LIBSVM_IMPL = ["c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"]
print('X_test_feature', X_test_feature.shape)
print('self.support_', clf.support_.shape, clf.support_)
print('self.support_vectors_', clf.support_vectors_.shape, np.array(clf.support_vectors_, dtype=np.float16))
print('self._n_support', clf._n_support)
print('self._dual_coef_', clf._dual_coef_.shape, np.array(clf._dual_coef_, dtype=np.float16))
print('self._intercept_', np.array(clf._intercept_, dtype=np.float16))
print('self._probA', clf._probA)
print('self._probB', clf._probB)
print('svm_type', LIBSVM_IMPL.index(clf._impl))
print('kernel', clf.kernel)
print('self.degree', clf.degree)
print('self.coef0', np.float16(clf.coef0))
print('self.gamma', np.float16(clf._gamma))
print('self.cache_size', clf.cache_size)
print()

np.savez('model/params.npz', support = clf.support_, SV = np.array(clf.support_vectors_, dtype=np.float16), nSV = clf._n_support, 
          sv_coef = np.array(clf._dual_coef_, dtype=np.float16), intercept = np.array(clf._intercept_, dtype=np.float16), svm_type = LIBSVM_IMPL.index(clf._impl), 
          kernel = clf.kernel, degree = clf.degree, gamma = np.float16(clf._gamma), coef0 = np.float16(clf.coef0))

print(clf.break_ties, clf.decision_function_shape, len(clf.classes_), clf._sparse, callable(clf.kernel))
print()

save_path = 'model/bin/'
clf.support_.tofile(save_path+'01_support_int32.bin')
clf.support_vectors_.astype(np.float32).tofile(save_path+'02_SV_float32.bin')
clf._n_support.tofile(save_path+'03_nSV_int32.bin')
clf._dual_coef_.astype(np.float32).tofile(save_path+'04_sv_coef_float32.bin')
clf._intercept_.astype(np.float32).tofile(save_path+'05_intercept_float32.bin')
np.array(LIBSVM_IMPL.index(clf._impl)).astype(np.float32).tofile(save_path+'06_svm_type_float32.bin')
np.array(clf.kernel).tofile(save_path+'07_kernel_int32_string.bin')
np.array(clf.degree).tofile(save_path+'08_degree_int32.bin')
np.array(clf._gamma).astype(np.float32).tofile(save_path+'09_gamma_float32.bin')
np.array(clf.coef0).astype(np.float32).tofile(save_path+'10_coef0_float32.bin')

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

start_time = time.time()
result = clf.predict(X_new_feature)
end_time = time.time()
print("{:f}s for {:d} new set predict".format(end_time - start_time, y_new.shape[0]))
print("{:d} positive classes in {:d} new set".format(np.sum(y_new), y_new.shape[0]))
print("accuracy_score: {:f}".format(accuracy_score(y_new, result)))
print("accuracy_number: {:d}/{:d}".format(int(accuracy_score(y_new, result, normalize=False)), len(y_new)))
print("f1_score: {:f}".format(f1_score(y_new, result)))
print()