##
## Created by Liu Deyuan on 2021/01/15.
##

from enum import EnumMeta
import numpy as np
import cv2
from util import fileTool as FT

if __name__ == '__main__':

    from svm_ldy import *
    from svm_origin import load_data, analysis_result

    import cv2
    from dataloader import read_bin, get_bin_file_with_width_and_height

    # a = '/Users/liudeyuan/Desktop/商汤杂项/SVM/input/rsz_warped_400x300.bin'
    # width, height = map(int, a.split('.')[0].split('_')[-1].split('x'))
    # image = read_bin(a, width, height)

    # a = '/Users/liudeyuan/Desktop/商汤杂项/SVM/input/rsz_warped.png'
    # image = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
    # image = np.array(image, dtype=np.float32)
    # image = image / 255.0

    with open('./test.txt', 'r') as f:
        lines = [i.strip() for i in f.readlines()]
    files = []
    labels = []
    for i, l in enumerate(lines):
        if i % 2 == 0:
            files.append(l[-6:])
        else:
            labels.append(int(l[-1:]))
    files = [f'/home/SENSETIME/liudeyuan1/Desktop/2TB/k1_result_0812/{i}/' for i in files]
    files = [FT.getAllFiles(i, ext='bin')[0] for i in files]
    labels = [1 if i == 7 else 0 for i in labels]

    images = []
    for a in files:
        width, height = map(int, a.split('.')[0].split('_')[-1].split('x'))
        image = read_bin(a, width, height)
        image = np.array(image, np.float32)
        image = cv2.resize(image, dsize=(400, 300), interpolation=cv2.INTER_LINEAR)
        image = np.array(image, np.float16)
        images.append(image)

#     image = np.array(image, np.float32)
#     # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     b = np.array(image * 255, dtype=np.uint8)
#     cv2.imwrite('1.png', b)

#     image = cv2.resize(image, dsize=(400, 300), interpolation=cv2.INTER_LINEAR)
#     image = np.array(image, np.float16)

#     images.append(image)
#     labels.append(1)
    X_test = np.array(images)
    y_test = np.array(labels)
    
    start_time = time.time()
    X_test_feature = feature_extraction(X_test)
    end_time = time.time()


    params = np.load('model/params.npz')
    support, SV, nSV, sv_coef, intercept, \
    svm_type, kernel, degree, gamma, coef0 = \
    params['support'], params['SV'], params['nSV'], \
    params['sv_coef'], params['intercept'], \
    params['svm_type'], params['kernel'], params['degree'], \
    params['gamma'], params['coef0']

    
    start_time = time.time()
    result = predict(X = X_test_feature, 
                     support = support,
                     SV = SV,
                     nSV = nSV, 
                     sv_coef = sv_coef,
                     intercept = intercept,  
                     svm_type = svm_type, 
                     kernel = kernel, 
                     degree = degree, 
                     gamma = gamma, 
                     coef0 = coef0)
    end_time = time.time()
    analysis_result(y_test, result, end_time-start_time)