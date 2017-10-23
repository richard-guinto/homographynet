'''
Author: Richard Guinto
Project: DeepHomography
Dependencies: keras
Usage: python <this file>
'''


import os.path
import glob

import numpy as np
from keras.models import load_model
#from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, InputLayer
from keras import backend as K
from keras import optimizers

def euclidean_l2(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

def mean_square_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1, keepdims=True)

# load model
model = load_model("/data/richard/model.hdf5", custom_objects={'euclidean_l2': euclidean_l2})

# Dataset-specific
predict_data_path = '/home/samsung/richard/dataset/test/f7823b11-901a-457a-89a2-8c37dc4fef60.npz'
samples_per_archive = 9216



archive = np.load(os.path.abspath(predict_data_path))
print('keys: ', archive.files)
images = archive['images']
offsets = archive['offsets']
sample_image = images[0];
sample_shape = sample_image.shape
print('sample shape: ', sample_shape)
sample_image = sample_image.reshape(1, 128, 128, 2)
#print('sample image shape: ', sample_image.shape)
sample_offset = offsets[0];
print('Sample Offset: ', sample_offset)
normalize_image = (sample_image - 127.5) / 127.5
normalize_offset = sample_offset / 32.


print('Normalize Sample Offset: ', normalize_offset)
offset = model.predict(normalize_image)
print('Predicted Nomralize Offset: ', offset)
pred_nonnorm_offset = offset * 32.
print('Predicted Non-normalize Offset: ', pred_nonnorm_offset)
smse = np.sqrt(np.sum(np.square(offset - normalize_offset),axis=-1,keepdims=True))
print('Normalize SMSE: ', smse)
mse = np.mean(np.square(offset - normalize_offset), axis=-1, keepdims=True)
print('Normalize MSE: ', mse)
nonnorm_mse = np.mean(np.square(pred_nonnorm_offset - sample_offset), axis=-1, keepdims=True)
print('Non-Normalize MSE: ', nonnorm_mse)
mae = np.mean(np.absolute(offset - normalize_offset), axis=-1, keepdims=True)
print('Normalize MAE: ', mae)
nonnorm_mae = np.mean(np.absolute(pred_nonnorm_offset - sample_offset), axis=-1, keepdims=True)
print('Non-Normalize MAE: ', nonnorm_mae)
sum = 0
for i in range(0, len(sample_offset),2):
    h = np.square(pred_nonnorm_offset[0][i] - sample_offset[i]) + np.square(pred_nonnorm_offset[0][i+1] - sample_offset[i+1])
    h = np.sqrt(h)
    print('h: ', h)
    sum = sum + h
sum = sum / (len(sample_offset) / 2)
print('average corner error: ', sum)

offset = model.predict(sample_image)
print('Predicted Offset: ', offset)
print('Target Offset: ', sample_offset)
smse = np.sqrt(np.sum(np.square(offset - sample_offset),axis=-1,keepdims=True))
print('SMSE: ', smse)
mse = np.mean(np.square(offset - sample_offset), axis=-1, keepdims=True)
print('MSE: ', mse)


