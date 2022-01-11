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
data = np.load('data_float16.npz')
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

# Extract the Feature
print('Begin Feature Extraction')
X_train_feature = feature_extraction(X_train, feature='HOG')
start_time = time.time()
X_test_feature = feature_extraction(X_test, feature='HOG')
end_time = time.time()
print("{:f}s for {:d} test feature extraction".format(end_time - start_time, y_test.shape[0]))

# print(np.isinf(X_train_feature).any())
# print(np.isfinite(X_train_feature).any())
# for i in range(len(X_train_feature)):
#     if np.isnan(X_train_feature[i]).any():
#         print(i)

# for x in X_train_feature:
#     x[np.isnan(x)] = 0
# for x in X_test_feature:
#     x[np.isnan(x)] = 0

# Construct the Model
print('Begin SVM')
clf = svm.SVC(gamma='scale', decision_function_shape='ovo')

# Train the Model
clf.fit(X_train_feature, y_train)

# Get the Result
start_time = time.time()
result = clf.predict(X_test_feature)
end_time = time.time()
print("{:f}s for {:d} tests predict".format(end_time - start_time, y_test.shape[0]))

# Print the Result with Evaluation
print("{:d} positive classes in {:d} tests".format(np.sum(y_test), y_test.shape[0]))
print("accuracy_score: {:f}".format(accuracy_score(y_test, result)))
print("accuracy_number: {:d}/{:d}".format(int(accuracy_score(y_test, result, normalize=False)), len(y_test)))
print("f1_score: {:f}".format(f1_score(y_test, result)))

# Followings are the test code, showing which sample is misclassified

# diff = y_test - result
#
# for i in range(y_test.shape[0]):
#     if diff[i] == 1:
#         cv2.imwrite('wa/FN/' + str(i) + '.png',
#                     (np.stack([X_test[i, :, :], X_test[i, :, :], X_test[i, :, :]], axis=2) * 255).astype(np.uint8))
#     elif diff[i] == -1:
#         cv2.imwrite('wa/FP/' + str(i) + '.png',
#                     (np.stack([X_test[i, :, :], X_test[i, :, :], X_test[i, :, :]], axis=2) * 255).astype(np.uint8))
