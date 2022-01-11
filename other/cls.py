import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from dataloader import dataloader
from utils import feature_extraction

# Load the Data, Class 0 and Class 1
X0, y0 = dataloader(0)
X1, y1 = dataloader(1)
# X1, y1 = X1 * 3, y1 * 3

# Complete Set X, y
X = np.concatenate((X0, X1))
y = np.concatenate((y0, y1))

# Split the Train Set and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract the Feature
print('Begin Feature Extraction')
X_train_feature = feature_extraction(X_train, feature='HOG')
X_test_feature = feature_extraction(X_test, feature='HOG')
# X_train_feature = feature_extraction(X_train, feature='LBP')
# X_test_feature = feature_extraction(X_test, feature='LBP')

# Construct the Model
print('Begin SVM')
clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
# print('Begin LR')
# clf = LogisticRegression(C=1000, class_weight='balanced', solver='liblinear', random_state=42)

# Train the Model
clf.fit(X_train_feature, y_train)

# Get the Result
result = clf.predict(X_test_feature)

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
