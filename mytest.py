import numpy as np
import cv2

from svm import *
from utils import feature_extraction

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

X_train_feature = np.array(X_train_feature)

# X_train_feature = featureNormalize(X_train_feature)
# print([np.mean(i) for i in X_train_feature])

times = []
numruns = 5
k = 1000

for i in range(numruns):
    print("Training model ", i + 1, ' out of ',
            numruns, "...")

    begin = time.time()
    mP = myPegasos(X_train_feature, y_train, 1e-4, k)

    end = time.time()
    times.append(end - begin)

    print('Runtime: ', round(end - begin, 3), ' seconds')

    print("\n")
# Print combined error rates for each train set
# percent averaged by the number of folds that ran
print("------FINAL RESULT -------")
print('Average runtime w/ minibatch size of ', k,
        ':\t', round(np.mean(times), 3), " sec.")

print('STD runtime w/ minibatch size of ', k,
        ':\t', round(np.std(times), 3), " sec.")