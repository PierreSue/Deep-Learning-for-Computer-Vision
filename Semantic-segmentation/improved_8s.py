import os 
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Conv2DTranspose, Cropping2D, Add
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras
import tensorflow as tf

label = 7

# crop o1 wrt o2
def crop( o1 , o2 , i  ):
    o_shape2 = Model( i  , o2 ).output_shape
    outputHeight2 = o_shape2[2]
    outputWidth2 = o_shape2[3]

    o_shape1 = Model( i  , o1 ).output_shape
    outputHeight1 = o_shape1[2]
    outputWidth1 = o_shape1[3]

    cx = abs( outputWidth1 - outputWidth2 )
    cy = abs( outputHeight2 - outputHeight1 )

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )))(o1)
    else:
        o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )))(o2)
                                                                        
    if outputHeight1 > outputHeight2 :
        o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )))(o1)
    else:
        o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )))(o2)

    return o1 , o2 

image_input = Input(shape=(512, 512, 3))

conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(image_input)
conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)

conv5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
conv5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
pool5 = MaxPooling2D(pool_size=(1, 1), strides=(2, 2))(conv5)

VGG = Model(image_input, pool5)
VGG.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

o = Conv2D(4096, (7,7), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool5)
o = Dropout(0.5)(o)
o = Conv2D(4096, (1,1), activation='relu', padding='same', kernel_initializer = 'he_normal')(o)
o = Dropout(0.5)(o)

o = Conv2D(label,(1,1), activation='linear', kernel_initializer='he_normal')(o)
o = Conv2DTranspose(label, kernel_size=(4,4), strides=(2,2), use_bias=False, activation='softmax', padding='same')(o)

o2 = pool4
o2 = Conv2D(label, (1,1), kernel_initializer='he_normal')(o2)
	
o, o2 = crop(o, o2, image_input)
o = Add()([o, o2])
o = Conv2DTranspose(label, kernel_size=(4,4), strides=(2,2), use_bias=False, activation='softmax', padding='same')(o)

o2 = pool3
o2 = Conv2D(label, (1,1), kernel_initializer='he_normal')(o2)

o2, o = crop(o2, o, image_input)
o = Add()([o2, o])
o = Conv2DTranspose(label, kernel_size=(16,16) ,  strides=(8,8) , use_bias=False, activation='softmax', padding='same')(o)

model = Model(input = image_input, output = o)
model.summary()
model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])


print("loading data")
val_sat = np.load('./data/val_sat.npy')
val_mask = np.load('./data/val_mask.npy')
train_sat = np.load('./data/train_sat.npy')
train_mask = np.load('./data/train_mask.npy')
print("loading data done")

model_name='FCN8s'
checkpointer = ModelCheckpoint(filepath='./model/'+model_name+'_model-{epoch:02d}-{val_loss:.4f}.h5'
	,verbose=0, save_best_only=False, period=1)
earlystopping = EarlyStopping(patience=15, min_delta=0.00)
model.fit(train_sat, train_mask, batch_size=15, epochs=200, verbose=1, 
          validation_data=(val_sat, val_mask),
          callbacks=[checkpointer, earlystopping])
model.save('./model/'+ model_name +'_model_final.h5')
