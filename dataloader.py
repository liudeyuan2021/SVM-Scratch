# This dataloader.py contains the necessary data loading and preprocess steps

from matplotlib import image
import numpy as np
import cv2
import os
from tqdm import tqdm
from util import fileTool as FT

data_dir = {
    0: '/home/SENSETIME/liudeyuan1/Desktop/2TB/oppo_result_0820/0', 
    1: '/home/SENSETIME/liudeyuan1/Desktop/2TB/oppo_result_0820/1',   
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
    for subdir in tqdm(subdirs):
        bin_file, width, height = get_bin_file_with_width_and_height(os.path.join(dir, subdir))
        image = read_bin(os.path.join(dir, subdir, bin_file), width, height)

        # torch.resize改成了opencv.resize，测试效果相同
        image = np.array(image, np.float32)
        image = cv2.resize(image, dsize=(400, 300), interpolation=cv2.INTER_LINEAR)
        # image = resize(torch.from_numpy(image)[None, None, :, :]).squeeze().numpy()

        # 再转回float16
        image = np.array(image, np.float16)

        images.append(image)
        labels.append(label)
    # print(len(images))
    # if label == 1:
    #     images = 5 * images
    #     labels = 5 * labels
    # print(len(images))
    return images, labels

def dataloader_other(label):
    """ Load the data from the label-specified directory, which is set in dict(data_dir) """
    images, labels = [], []

    files = FT.getAllFiles(data_dir[label])
    files = [i for i in files if not i.count('._')]
    for file in tqdm(files):
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(400, 300), interpolation=cv2.INTER_LINEAR)
        image = np.array(image, dtype=np.float16)
        images.append(image)
        labels.append(label)

    return images, labels

if __name__ == '__main__':
    dataloader(1)

