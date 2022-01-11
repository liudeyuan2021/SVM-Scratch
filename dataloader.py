# This dataloader.py contains the necessary data loading and preprocess steps

import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
from torch.nn import UpsamplingBilinear2d

data_dir = {
    0: 'dataset/0',    # Class 0 with its Data_Dir
    1: 'dataset/1',    # Class 1 with its Data_Dir
}


def read_bin(path, width, height):
    """ Read bin file from path with width and height specifically """
    return np.fromfile(path, dtype=np.float16).reshape(height, width)  # .astype(np.float32)


def get_bin_file_with_width_and_height(path):
    """ Input: A path with only one bin file inside, whose filename is like *_*_{w}x{h}.bin
        Output: bin_file's name and its width and height
    """
    files = os.listdir(path)
    bin_file, width, height = None, None, None
    for file in files:
        if os.path.splitext(file)[1] == '.bin':
            bin_file = file
            width, height = map(int, bin_file.split('.')[0].split('_')[-1].split('x'))
            break
    return bin_file, width, height


def dataloader(label):
    """ Load the data from the label-specified directory, which is set in dict(data_dir) """
    images, labels = [], []

    dir = data_dir[label]
    subdirs = os.listdir(dir)
    resize = UpsamplingBilinear2d(size=(300, 400))
    # for subdir in subdirs:
    for subdir in tqdm(subdirs):
        bin_file, width, height = get_bin_file_with_width_and_height(os.path.join(dir, subdir))
        # print(bin_file, width, height, sep=' ')
        image = read_bin(os.path.join(dir, subdir, bin_file), width, height)
        # image = cv2.resize(image, (400, 300))
        image = resize(torch.from_numpy(image)[None, None, :, :]).squeeze().numpy()
        # print(image.shape)
        images.append(image)
        labels.append(label)

    return images, labels

def dataloader_test(label):
    images, labels = [], []

    dir = data_dir[label]
    files = os.listdir(dir)
    resize = UpsamplingBilinear2d(size=(300, 400))
    
    for file in tqdm(files):
        image = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (400, 300)).squeeze()
        image = np.array(image, dtype=np.float)
        image = image / 255.0
        # print(image)
        # cv2.imshow('0', image)
        # cv2.waitKey(1000)
        # image = resize(torch.from_numpy(image)[None, None, :, :]).squeeze().numpy()
        images.append(image)
        labels.append(label)

    return images, labels

if __name__ == '__main__':
    dataloader_test(0)

