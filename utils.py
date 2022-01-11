import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import hog


def feature_extraction(arr, feature='LBP'):
    """ Return a list of feature vectors of the arr, each image is arr[i] """
    assert feature in ['LBP', 'HOG', 'HU']
    arr_feature = []
    if feature == 'LBP':
        for i in range(arr.shape[0]):
            arr_feature.append(extract_lbp_feature(arr[i]))
        return arr_feature
    elif feature == 'HOG':
        for i in range(arr.shape[0]):
            arr_feature.append(extract_hog_feature(arr[i]))
        return arr_feature
    elif feature == 'HU':
        for i in range(arr.shape[0]):
            arr_feature.append(np.append(extract_lbp_feature(arr[i]), extract_hu_moments(arr[i])))
        return arr_feature


def extract_lbp_feature(image):
    """ Return the lbp feature histogram vector """
    # lbp = local_binary_pattern(image, 24, 8, method="uniform")
    # (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp = local_binary_pattern(image, 8, 1, method="uniform")
    n_bins = int(lbp.max() + 1)
    (hist, _) = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_hog_feature(image):
    """ Return the hog feature vector """
    hog_feat = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')
    return hog_feat


def extract_hu_moments(image):
    """ Return the Hu Moments vector """
    _, image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    hu_moments = hu_moments.flatten()
    return hu_moments
