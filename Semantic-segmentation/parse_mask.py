import os
import sys
import numpy as np
import scipy.misc as misc
import argparse

label = 7
def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.zeros((n_masks, 512, 512, label), dtype=np.uint8)

    for i, file in enumerate(file_list):
        mask = misc.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = [1,0,0,0,0,0,0]  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = [0,1,0,0,0,0,0]  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = [0,0,1,0,0,0,0]  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = [0,0,0,1,0,0,0]  # (Green: 010) Forest land 
        masks[i, mask == 1] = [0,0,0,0,1,0,0]  # (Blue: 001) Water 
        masks[i, mask == 7] = [0,0,0,0,0,1,0]  # (White: 111) Barren land 
        masks[i, mask == 0] = [0,0,0,0,0,0,1]  # (Black: 000) Unknown 

    return masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--ground-truth', help='ground truth masks', type=str)
    parser.add_argument('-p', '--predict', help='predict masks', type=str)
    FLAGS = parser.parse_args()

    ground = read_masks(FLAGS.ground_truth)
    np.save('data/ground_truth.npy', ground)
    
    predict = read_masks(FLAGS.predict)
    np.save('data/predict.npy', predict)
    
