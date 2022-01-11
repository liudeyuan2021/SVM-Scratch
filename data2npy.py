# This data2npy.py is to preload the data and save to the hard-disk with extension name .npz
# To do this, we can save the time of data pre-processing, so save the time for debugging

import numpy as np
from sklearn.model_selection import train_test_split
# from dataloader import dataloader
from dataloader import dataloader_test

# X0, y0 = dataloader(0)
# X1, y1 = dataloader(1)
X0, y0 = dataloader_test(0)
X1, y1 = dataloader_test(1)
# X1, y1 = X1 * 3, y1 * 3

# Complete Set X, y
X = np.concatenate((X0, X1))
y = np.concatenate((y0, y1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.savez('dataset/data_float16.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)