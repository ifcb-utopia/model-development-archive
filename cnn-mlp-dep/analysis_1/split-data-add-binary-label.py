'''
    Add first exclusion flag for missing metadata & EDSA threshold: 
        I: add exclusion flag
        II: add train / test flag (in this case get from existing split)
        III: add binary label flag 
'''


# -- 
# dependancies 

import os 
import pandas as pd 
import numpy as np 


# -- 
# io 

data = pd.read_csv('./reference_data/image-file-directory.csv')


# -- 
# add exclusion flag 

def excluded_1(row): 
    if row['ESDA_exclude'] == True: 
        out = 1
    elif row['missing_meta_data'] == True: 
        out = 1test
    else: 
        out = 0
    return out

data['excluded_1'] = data.apply(lambda row: excluded_1(row), axis=1)


# -- 
# code for splitting data (run in future analyses)

# data_test = data.sample(frac=0.15)
# data_train = data.loc[~data['file_name_x'].isin(data_test.file_name_x)]
# data_test['set'] = 'test'
# data_train['set'] = 'train'
# data_split = pd.concat([data_train, data_test])


# -- 
# getting testing and training data flags from summary files 

test_data = pd.read_csv('/Users/culhane/Desktop/plankton-vision/model-one-evaluation-data-w-flags.csv')
test_data['set'] = 'test'
test_data_sub = test_data[['file_name_x', 'set']]
test_data_sub.columns = ['file_name', 'set']

data = pd.merge(data, test_data_sub, on='file_name', how='left')

def coerce_set(row): 
    if isinstance(row['set'], float): 
        out = 'train'
    else: 
        out = 'test'
    return out

data['set'] = data.apply(lambda row: coerce_set(row), axis=1)

''' check set assignment since we are using retroactive wonky method '''
data.groupby('set').agg({'file_name' : 'count'})

#        file_name
# set             
# test      299835
# train    1699065


# -- 
# add binary label flag 

def binary_label(row): 
    if row['high_group'] in ['Multiple', 'Corrupt', 'Not living']: 
        out = 'not_plankton'
    else: 
        out = 'plankton'
    return out


data['binary_label'] = data.apply(lambda row: binary_label(row), axis=1)

''' check this assignment '''
data.groupby('binary_label').agg({'file_name' : 'count'})


# -- 
# write out data for labels 

data.to_csv('./reference_data/image-file-directory.csv', index=False)

