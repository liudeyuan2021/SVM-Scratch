import numpy as np
import cv2

if __name__ == '__main__':
    file = 'dataset/0/cat.4001.jpg'    
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    img = np.fromfile(file, dtype=np.float16)
    print(img.shape)