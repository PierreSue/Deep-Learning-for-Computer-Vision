import sys
import os 
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
import scipy.misc
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras
import tensorflow as tf
import argparse

label = 7

def seven_to_three(val, seven_image):
    masks = np.empty((len(seven_image), 512, 512,3))
    for id7 ,img7 in enumerate(seven_image):
        for idr, row in enumerate(img7):
            for idc, pixel in enumerate(row):
                index = np.argmax(pixel)
                if index == 0:
                    masks[id7, idr, idc] = [0,1,1]
                elif index == 1:
                    masks[id7, idr, idc] = [1,1,0]
                elif index == 2:
                    masks[id7, idr, idc] = [1,0,1]
                elif index == 3:
                    masks[id7, idr, idc] = [0,1,0]
                elif index == 4:
                    masks[id7, idr, idc] = [0,0,1]
                elif index == 5:
                    masks[id7, idr, idc] = [1,1,1]
                elif index == 6:
                    masks[id7, idr, idc] = [0,0,0]
    return masks


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-name', help='model(.h5 file)', type=str)
parser.add_argument('-o', '--output-dir', help='output directory', type=str)

FLAGS = parser.parse_args()
model = load_model(FLAGS.model_name)

print("loading data")
test_sat = np.load('./data/test_sat.npy')
print(test_sat.shape)
print("loading data done")

for id, x in enumerate(test_sat):
    val_image = []
    val_image.append(x)
    val_image = np.array(val_image)
    predicted_image = model.predict(val_image, batch_size=32, verbose=0, steps=None)
    print('picture_', id, ': ', predicted_image.shape)
    images = seven_to_three(test_sat,predicted_image)
    strid = str(id).zfill(4)
    scipy.misc.imsave(FLAGS.output_dir+'/'+strid+'_mask.png', images[0])
