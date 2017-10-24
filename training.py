'''
Author: Richard Guinto
Project: DeepHomography
Dependencies: keras
Usage: python <this file>
'''


# From baudm
# Efficient loading in Keras using a Python generator

import os.path
import glob

import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, InputLayer
from keras.callbacks import ModelCheckpoint, ProgbarLogger
from keras import backend as K
from keras import optimizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def data_loader(path, batch_size=64):
    """Generator to be used with model.fit_generator()"""
    while True:
        for npz in glob.glob(os.path.join(path, '*.npz')):
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']
            offsets = archive['offsets']
            # Yield minibatch
            for i in range(0, len(offsets), batch_size):
                end_i = i + batch_size
                try:
                    batch_images = images[i:end_i]
                    batch_offsets = offsets[i:end_i]
                except IndexError:
                    continue
                # Normalize
                batch_images = (batch_images - 127.5) / 127.5
                batch_offsets = batch_offsets / 32.
                yield batch_images, batch_offsets
#end of baudm code

def euclidean_l2(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))
#    return K.sqrt(K.maximum(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True), K.epsilon()))


# Dataset-specific
train_data_path = '/home/samsung/richard/dataset/training'
test_data_path = '/home/samsung/richard/dataset/test'
samples_per_archive = 9216
num_archives = 40
num_samples = num_archives * samples_per_archive # 43 archives x 9,216 samples per archive, but use just 40 and save the 3 for testing

# From the paper
batch_size = 64
total_iterations = 90000

steps_per_epoch = num_samples / batch_size # As stated in Keras docs
epochs = int(total_iterations / steps_per_epoch)

input_shape = (128, 128, 2)
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.5


model = Sequential()
model.add(InputLayer(input_shape))
model.add(Conv2D(filters=filters,\
        kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters,\
        kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,\
        kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters,\
        kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters*2,\
        kernel_size=kernel_size, activation='relu', padding='same',))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters*2,\
        kernel_size=kernel_size, activation='relu', padding='same',))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters*2,\
        kernel_size=kernel_size, activation='relu', padding='same',))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters*2,\
        kernel_size=kernel_size, activation='relu', padding='same',))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(dropout))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(dropout))
#for regression model
model.add(Dense(8))
#model.add(Activation('softmax'))
model.summary()

#use optimizer Stochastic Gradient Methond with a Learning Rate of 0.005 and momentum of 0.9
#sgd = optimizers.SGD(lr=0.005, momentum=0.9, decay=0.001355)
sgd = optimizers.SGD(lr=0.005, momentum=0.9)

#compile model
model.compile(loss=euclidean_l2,\
        optimizer=sgd, metrics=['mean_squared_error'])

#check point
filepath = "/data/richard/20171024/checkpoint-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1)
callback_list = [checkpoint]


# Train
print('TRAINING...')
model.fit_generator(data_loader(train_data_path, batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs, callbacks=callback_list)

print('TESTING...')
# Test
score = model.evaluate_generator(data_loader(test_data_path, batch_size),
                         steps=3*samples_per_archive/batch_size)

print('score: ', score)
