# This data2npy.py is to preload the data and save to the hard-disk with extension name .npz
# To do this, we can save the time of data pre-processing, so save the time for debugging

import numpy as np
from sklearn.model_selection import train_test_split
from dataloader import dataloader, data_dir

X, y = [], []

for i in data_dir:
    X_part, y_part = dataloader(i)
    X.append(X_part)
    y.append(y_part)

X = np.concatenate(X)
y = np.concatenate(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.savez('dataset/data_float16_v3.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)