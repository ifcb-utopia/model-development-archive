'''
    functions to load data for models: 
        I: multi-input binary classification network
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


# -- 
# I: multi-input binary model data generators and utilities 


''' data serialization and augmentation functions '''

def preprocess_input(image):
    fixed_size = 128
    image_size = image.shape[:2] 
    ratio = float(fixed_size)/max(image_size)
    new_size = tuple([int(x*ratio) for x in image_size])
    img = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = fixed_size - new_size[1]
    delta_h = fixed_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    rescaled_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return(rescaled_image)

def multi_stream_generator(dataset, path, batch_size, lb):
    data_size = len(dataset)
    n_batches = data_size / batch_size
    remain = data_size % batch_size 
    # lb = None
    while True: 
        files = dataset.sample(n=data_size - remain)
        shuffled = files.sample(frac=1)
        result = np.array_split(shuffled, n_batches)  
        for batch in result: 
            labels = batch['label'].values
            # if lb == None: 
            #     lb = LabelBinarizer()
            labels = lb.fit_transform(labels)
            vector_data = batch.drop(['file_name_x', 'label'], axis=1).values
            image_data = []
            for i in range(len(batch)): 
                row = batch.iloc[i]
                input_path = path + row['label'] + '/' + row['file_name_x']
                image_data.append(preprocess_input(cv2.imread(input_path)))
                # 
            image_data = np.array(image_data)
            yield ([ image_data, vector_data ] , labels )

def multi_stream_generator_final(dataset, path, batch_size, lb):
    data_size = len(dataset)
    n_batches = data_size / batch_size
    remain = data_size % batch_size 
    # lb = None
    while True: 
        files = dataset.sample(n=data_size - remain)
        shuffled = files.sample(frac=1)
        result = np.array_split(shuffled, n_batches)  
        for batch in result: 
            labels = batch['label'].values
            # if lb == None: 
            #     lb = LabelBinarizer()
            labels = lb.fit_transform(labels)
            vector_data = batch.drop(['file_name_x', 'label'], axis=1).values
            image_data = []
            for i in range(len(batch)): 
                row = batch.iloc[i]
                input_path = path + row['file_name_x']
                image_data.append(preprocess_input(cv2.imread(input_path)))
                # 
            image_data = np.array(image_data)
            yield ([ image_data, vector_data ] , labels )


def multi_stream_generator_SLC(dataset, path, batch_size, lb):
    data_size = len(dataset)
    n_batches = data_size / batch_size
    remain = data_size % batch_size 
    # lb = None
    while True: 
        files = dataset.sample(n=data_size - remain)
        shuffled = files.sample(frac=1)
        result = np.array_split(shuffled, n_batches)  
        for batch in result: 
            labels = batch['label'].values
            # if lb == None: 
            #     lb = LabelBinarizer()
            labels = lb.transform(labels)
            vector_data = batch.drop(['file_name_x', 'label'], axis=1).values
            image_data = []
            for i in range(len(batch)): 
                row = batch.iloc[i]
                input_path = path + row['file_name_x']
                image_data.append(preprocess_input(cv2.imread(input_path)))
                # 
            image_data = np.array(image_data)
            yield ([ image_data, vector_data ] , labels )


''' image sampling / loading and directory structuring utilities '''

def clean_sample_directory(data_path): 
    classes = [c for c in os.listdir(data_path) if c != '.DS_Store']
    for c in classes: 
        os.rmdir(data_path + c)

def move_files_back_home(data_path): 
    plankton_files_home = '/Users/culhane/Desktop/NAAMES/'
    classes = [c for c in os.listdir(data_path) if c != '.DS_Store']
    files = []
    for c in classes:
        files_c = [{'whole_path' : data_path + c + '/' + i, 'file_name_x' : i} for i in os.listdir(data_path + c + '/')]
        files.extend(files_c)
    df_files= pd.DataFrame(files)
    df_files['to_path'] = df_files.apply(lambda row: plankton_files_home + row['file_name_x'], axis=1)
    for i in range(len(df_files)): 
        info = df_files.iloc[i]
        os.rename(info['whole_path'], info['to_path'])
    clean_sample_directory(data_path)

def build_image_directory(vector_data):
    classes = vector_data.label.unique().tolist()
    for _set in ['train', 'test', 'validation']: 
        if not os.path.exists('./data/' + _set): 
            os.mkdir('./data/' + _set)
        for c in classes: 
            os.mkdir('./data/' + _set + '/' + c)

def move_sampled_images(data_sample, data_name):
    plankton_files_home = '/Users/culhane/Desktop/NAAMES/'
    path = './data/' + data_name + '/'
    missed_files = []
    for i in range(len(data_sample)): 
        try: 
            row = data_sample.iloc[i]
            to_path = path + row['label'] + '/' + row['file_name_x']
            from_path = plankton_files_home + row['file_name_x']
            to_path_sub = path + row['label']
            if not os.path.exists(from_path): 
                print(from_path)
            if not os.path.exists(to_path_sub): 
                print(to_path_sub)
            os.rename(from_path, to_path)
        except: 
            missed_files.append(data_sample.iloc[i]['file_name_x'])
    return missed_files







