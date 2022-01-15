##
## Created by Liu Deyuan on 2021/01/15.
##

import numpy as np
from skimage.feature import hog


def feature_extraction(arr):
    arr_feature = []
    for i in range(arr.shape[0]):
        arr_feature.append(hog(arr[i], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys'))
    return np.array(arr_feature)

