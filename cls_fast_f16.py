import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from dataloader import dataloader
from utils import feature_extraction

import time

# Load the Data which is Preloaded by data2npy.py
data = np.load('dataset/data_float16.npz')
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

# Extract the Feature
print('Begin Feature Extraction')
X_train_feature = feature_extraction(X_train, feature='HOG')
start_time = time.time()
X_test_feature = feature_extraction(X_test, feature='HOG')
end_time = time.time()
print("{:f}s for {:d} test feature extraction".format(end_time - start_time, y_test.shape[0]))
print()

X_train_feature = np.array(X_train_feature)
X_test_feature = np.array(X_test_feature)

# Construct the Model
print('Begin SVM')
clf = svm.SVC()

# Train the Model
clf.fit(X_train_feature, y_train)

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

# Print the Result with Evaluation
start_time = time.time()
result = clf.predict(X_train_feature)
end_time = time.time()
print("{:f}s for {:d} tests predict".format(end_time - start_time, y_train.shape[0]))
print("{:d} positive classes in {:d} tests".format(np.sum(y_train), y_train.shape[0]))
print("accuracy_score: {:f}".format(accuracy_score(y_train, result)))
print("accuracy_number: {:d}/{:d}".format(int(accuracy_score(y_train, result, normalize=False)), len(y_train)))
print("f1_score: {:f}".format(f1_score(y_train, result)))
print()

start_time = time.time()
result = clf.predict(X_test_feature)
end_time = time.time()
print("{:f}s for {:d} tests predict".format(end_time - start_time, y_test.shape[0]))
print("{:d} positive classes in {:d} tests".format(np.sum(y_test), y_test.shape[0]))
print("accuracy_score: {:f}".format(accuracy_score(y_test, result)))
print("accuracy_number: {:d}/{:d}".format(int(accuracy_score(y_test, result, normalize=False)), len(y_test)))
print("f1_score: {:f}".format(f1_score(y_test, result)))
print()