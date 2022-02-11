'''
    functions to build models: 
        I: multi-input binary classification model 
'''

# -- 
# dependancies 

import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax

import re
import os
import cv2
import argparse
import locale

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from keras import regularizers


# --
# I: multi-input binary classification model 


''' MLP network architectures '''

def create_mlp(dim, regress=False):
    # define our MLP network
    # model = Sequential()
    # model.add(Dense(8, input_dim=dim, activation="relu"))
    # model.add(Dense(4, activation="relu"))
    model = Sequential()
    model.add(Dense(12, input_dim=dim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
    # return our model
    return model

def create_mlp_model1b_l2(dim, regress=False):
    # define our MLP network
    # model = Sequential()
    # model.add(Dense(8, input_dim=dim, activation="relu"))
    # model.add(Dense(4, activation="relu"))
    l2_param = 0.0001
    print('adding l2 norm penalty to MLP network')
    model = Sequential()
    model.add(Dense(12, input_dim=dim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, kernel_regularizer=regularizers.l2(l2_param), activation='relu'))
    # model.add(Dense(1000, activation='relu'))    
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
    # return our model
    return model


def create_mlp_model1b(dim, regress=False):
    # define our MLP network
    # model = Sequential()
    # model.add(Dense(8, input_dim=dim, activation="relu"))
    # model.add(Dense(4, activation="relu"))
    model = Sequential()
    model.add(Dense(12, input_dim=dim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    # model.add(Dense(1000, activation='relu'))    
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
    # return our model
    return model

def create_mlp_model_A1(dim, regress=False):
    # define our MLP network
    # model = Sequential()
    # model.add(Dense(8, input_dim=dim, activation="relu"))
    # model.add(Dense(4, activation="relu"))
    model = Sequential()
    model.add(Dense(12, input_dim=dim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(1000, activation='relu'))    
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
    # return our model
    return model


''' convolutional network architectures '''

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model


def create_cnn_deeper(width, height, depth, filters=(16, 64, 32, 128, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model


def create_cnn_model1b(width, height, depth, filters=(32, 32, 64, 128, 256, 256), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    print('doing the correct script again: dropout 20%')
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(rate=0.5)(x)
        x = Dropout(rate=0.2)(x)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    # x = Dropout(rate=0.5)(x)
    x = Dropout(rate=0.2)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(512)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model


''' What activation function do we want to have after the last fully connected layer given that we arent making preds then '''

def create_cnn_model_A1(width, height, depth, filters=(32, 16, 64, 32, 128, 128, 64, 256, 256, 128), regress=False):
    print('doing the correct script again: dropout 20%')
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        if i in [1, 3, 6, 9]:
            x = MaxPooling2D(pool_size=(3, 3))(x)
            # x = Dropout(rate=0.5)(x)
            # x = Dropout(rate=0.2)(x)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    # x = Dropout(rate=0.5)(x)
    x = Dropout(rate=0.2)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(1000)(x)
    x = Activation("relu")(x)
    # x = Dropout(rate=0.2)(x)
    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model




