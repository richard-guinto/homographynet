'''
Author: Richard Guinto
Project: DeepHomography
Dependencies: keras
Usage: python <this file>
'''


import os.path
#import glob

import numpy as np
from keras.models import load_model
#from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, InputLayer
from keras import backend as K
from keras import optimizers

def euclidean_l2(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

def mean_square_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1, keepdims=True)

#checkpoint
checkpoint = "/data/richard/model.hdf5"

# Dataset-specific
predict_data_path = '/home/samsung/richard/dataset/test/f7823b11-901a-457a-89a2-8c37dc4fef60.npz'
samples_per_archive = 9216



# load model
print('Loading model... ', checkpoint)
model = load_model(checkpoint, custom_objects={'euclidean_l2': euclidean_l2})

print('Loading archive... ', os.path.abspath(predict_data_path))
archive = np.load(os.path.abspath(predict_data_path))
print('keys: ', archive.files)
images = archive['images']
offsets = archive['offsets']

for idx in range(10):
    print('*************** SAMPLE ', idx)
    sample_image = images[idx];
    sample_shape = sample_image.shape
    sample_image = sample_image.reshape(1, 128, 128, 2)
    sample_offset = offsets[idx];
    print('Sample Offset: ', sample_offset)
    norm_sample_image = (sample_image - 127.5) / 127.5
    norm_sample_offset = sample_offset / 32.
    print('Normalize Sample Offset: ', norm_sample_offset)

    print('Predicting Offset...')
    norm_pred_offset = model.predict(norm_sample_image)
    print('Predicted Offset(Normalize): ', norm_pred_offset)
    pred_offset = norm_pred_offset * 32.
    print('Predicted Offset: ', pred_offset)
    norm_rmse = np.sqrt(np.sum(np.square(norm_pred_offset - norm_sample_offset),axis=-1,keepdims=True))
    print('Normalize RMSE: ', norm_rmse)
    norm_mse = np.mean(np.square(norm_pred_offset - norm_sample_offset), axis=-1, keepdims=True)
    print('Normalize MSE: ', norm_mse)
    mse = np.mean(np.square(pred_offset - sample_offset), axis=-1, keepdims=True)
    print('MSE: ', mse)
    norm_mae = np.mean(np.absolute(norm_pred_offset - norm_sample_offset), axis=-1, keepdims=True)
    print('Normalize MAE: ', norm_mae)
    mae = np.mean(np.absolute(pred_offset - sample_offset), axis=-1, keepdims=True)
    print('MAE: ', mae)
    sum = 0
    for i in range(0, len(sample_offset),2):
        h = np.square(pred_offset[0][i] - sample_offset[i]) + np.square(pred_offset[0][i+1] - sample_offset[i+1])
        h = np.sqrt(h)
        print('h: ', h)
        sum = sum + h
    sum = sum / (len(sample_offset) / 2)
    print('Ave. Corner Error: ', sum)



