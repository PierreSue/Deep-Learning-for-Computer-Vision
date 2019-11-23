import sys
import os 
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Conv2DTranspose
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras
from data import *

label = 7

image_input = Input(shape=(512, 512, 3))

conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(image_input)
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1)
pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2)
pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3)
conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3)
pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv4)
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv4)
pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv5)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv5)
pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)

VGG = Model(image_input, pool5)
VGG.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

conv6 = Conv2D(4096, (2,2), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool5)
drop6 = Dropout(0.5)(conv6)
conv7 = Conv2D(4096, (1,1), activation='relu', padding='same', kernel_initializer = 'he_normal')(drop6)
drop7 = Dropout(0.5)(conv7)
conv8 = Conv2D(label, (1,1), activation='linear', padding='valid', kernel_initializer='he_normal')(drop7)

Tconv8 = Conv2DTranspose(label, kernel_size=64, strides=32, use_bias=False, activation='softmax', padding='same')(conv8)

model = Model(image_input, Tconv8)
model.summary()
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print("loading data")
val_sat = np.load('./data/val_sat.npy')
val_mask = np.load('./data/val_mask.npy')
train_sat = np.load('./data/train_sat.npy')
train_mask = np.load('./data/train_mask.npy')
print("loading data done")

model_name = 'FCN32s'
earlystopping = EarlyStopping(patience=15, min_delta=0.00)
checkpointer = ModelCheckpoint(filepath='./model/'+ model_name +'_model-{epoch:02d}-{val_loss:.4f}.h5'
	, verbose=0, save_best_only=False, period=1)
model.fit(train_sat, train_mask, batch_size=15, epochs=200, 
	verbose=1, validation_data=(val_sat, val_mask),
	callbacks=[checkpointer, earlystopping])
model.save('./model/'+ model_name +'_model_final.h5')
