
'''
    Produce samples of different sources of error 
'''

# -- 
# dependancies 

import os
import pandas as pd
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from pprint import pprint
import re

# -- 
# io 

data = pd.read_csv('./testing-data-evaluation-for-review.csv')


# -- 
# get samples for binary data misclasifications (2 classes: should we do for worst groups?)

''' plankton error: plankton --> not plankton misclasification '''

total_error = data.loc[(data['excluded_1'] == 0) & (data['is_correct_bc'] == 0)] # 8468
error_1 = data.loc[(data['excluded_1'] == 0) & (data['is_correct_bc'] == 0) & (data['label_bc'] == 'plankton')] # 5803
error_1a = error_1.loc[error_1['higher_level_group'] == 'Other'] # 4875 (200) **
error_1b = error_1.loc[error_1['higher_level_group'] != 'Other'] # 928 (100) **
error_2 = data.loc[(data['excluded_1'] == 0) & (data['is_correct_bc'] == 0) & (data['label_bc'] == 'not_plankton') & (data['is_correct_mc'] == 0)] # (100) ** 2364 (incor mc also), 2683 (raw)


# -- 
# get samples for mutliclass misclassification (should we do for worst groups?)

total_error2 = data.loc[(data['excluded_2'] == 0) & (data['is_correct_mc'] == 0)] # 26368
error2_1 = data.loc[(data['excluded_2'] == 0) & (data['is_correct_mc'] == 0) & (~data['label_mc'].isin(['Other', 'Not_plankton']))] # 4487
error2_1a = error2_1.loc[error2_1['pred_label_mc_string'] == 'Other'] # 2885 (2951 w Not plankton) (100) **
error2_1b = error2_1.loc[~error2_1['pred_label_mc_string'].isin(['Other', 'Not_plankton'])] # 1536 (100) **
error2_2 = data.loc[(data['excluded_1'] == 0) & (data['is_correct_mc'] == 0) & (data['label_mc'] == 'Other') ] # 19517 
error2_2a = data.loc[(data['excluded_1'] == 0) & (data['is_correct_mc'] == 0) & (data['label_mc'] == 'Other') & (data['pred_label_mc_string'] != 'Not_plankton')] # 18969 (300) **
error2_2b = data.loc[(data['excluded_1'] == 0) & (data['is_correct_mc'] == 0) & (data['label_mc'] == 'Other') & (data['pred_label_mc_string'] == 'Not_plankton')] # 548
error2_3 = data.loc[(data['excluded_1'] == 0) & (data['is_correct_mc'] == 0) & (data['label_mc'] == 'Not_plankton')] # 2364


# -- 
# consolidate image samples for evaluation 

binary_error = pd.concat([error_1a.sample(n=200), error_1b.sample(n=100), error_2.sample(n=100)]) 
multiclass_error = pd.concat([error2_1a.sample(n=100), error2_1b.sample(n=100), error2_2a.sample(n=300)])

# len(binary_error) + len(multiclass_error) # 31, 876
# total_images_misclass = pd.concat([total_error, total_error2])
# len(total_images_misclass.file_name.unique()) # 32,490

error_samples = pd.concat([binary_error, multiclass_error])
error_samples['path'] = '/Users/culhane/Desktop/NAAMES/test/'

# -- 
# transform images and write out for review 

def get_image(row):
    input_path = row['path'] + row['file_name']
    return preprocess_input(cv2.imread(input_path, 2))

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



# -- 
# test write out single image 

row = error_samples.iloc[0]

img = get_image(row)
cv2.imwrite('/Users/culhane/Desktop/test-img.png', img)


# -- 
# make directory for errors and save all images as they are named in metadata 

def get_id(row): 
    fn = row['file_name']
    return re.match('(.*?)\_', fn).groups()[0]

error_samples['file_name_out'] = error_samples.apply(lambda row: get_id(row), axis=1)

error_samples = error_samples.sample(frac=1)


error_dir = '/Users/culhane/Desktop/image-errors/'
os.mkdir(error_dir)

for i in range(len(error_samples)): 
    row = error_samples.iloc[i]
    img = get_image(row)
    name = row['file_name_out']
    cv2.imwrite(error_dir + name + '.png', img)













