##
## Created by Liu Deyuan on 2021/01/15.
##

import numpy as np
import cv2
from util import fileTool as FT

if __name__ == '__main__':

        from dataloader import read_bin, get_bin_file_with_width_and_height

        a = '/home/SENSETIME/liudeyuan1/Downloads/l2b_result_0520/20220316210730/warp_merged_4096x3072.bin'

        images = []
        labels = []

        width, height = map(int, a.split('.')[0].split('_')[-1].split('x'))
        image = read_bin(a, width, height)

        image = np.array(image, np.float32)
        # image = cv2.resize(image, dsize=(400, 300), interpolation=cv2.INTER_LINEAR)
        # image = np.array(image, np.float16)

        # images.append(image)
        # labels.append(2)
        # X_test = np.concatenate(images)
        # y_test = np.concatenate(labels)


        image = np.array(image * 255, dtype=np.uint8)
        cv2.imwrite('4.png', image)
        print(image.shape)