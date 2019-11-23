import os
import sys
import numpy as np
import scipy.misc as misc
import argparse

label = 7
def read_image(filepath):
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    num = len(file_list)
    images = np.zeros((num, 512, 512, 3), dtype=np.uint8)
    for id, file in enumerate(file_list):
        image = misc.imread(os.path.join(filepath, file))
        images[id] = image

    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', help='test data', type=str)
    FLAGS = parser.parse_args()

    test_image = read_image(FLAGS.test)

    np.save('data/test_sat.npy', test_image)
    
