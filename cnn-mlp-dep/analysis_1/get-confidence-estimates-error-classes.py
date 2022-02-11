'''
    Review manual validation of misclassifications from analysis 1
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

data = pd.read_csv('./reference_data/testing-data-evaluation-for-review.csv')
labeled = pd.read_csv('./reference_data/labeled_image_errors.csv')

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
# add error flags for different subsets 

# error_1a --> other to not plankton 
# error_1b --> non other plankton to not plankton 
# error_2 --> not plankton to other / plankton 

# error2_1a --> plankton to other 
# error2_1b --> plankton to different plankton
# error2_2a --> other to plankton 

error_1a['e_class'] = 'other_to_not_plankton_bc'
error_1b['e_class'] = 'non_other_plankton_to_not_plankton_bc'
error_2['e_class'] = 'not_plankton_to_plankton_bc'
error2_1a['e_class'] = 'plankton_to_other_mc'
error2_1b['e_class'] = 'plankton_to_non_other_plankton_mc'
error2_2a['e_class'] = 'other_to_plankton_mc'

# -- 
# consolidate image samples for evaluation 
binary_error = pd.concat([error_1a, error_1b, error_2]) 
multiclass_error = pd.concat([error2_1a, error2_1b, error2_2a])
binary_error['model'] = 'BC'
multiclass_error['model'] = 'MC'
error_classes = pd.concat([binary_error, multiclass_error])

def get_id(row): 
    fn = row['file_name']
    return re.match('(.*?)\_', fn).groups()[0]

error_classes['id'] = error_classes.apply(lambda row: get_id(row), axis=1)


# -- 
# subset and join to corrected labels file 

labeled.columns = ['id', 'coerced_label']
eval_data = pd.merge(labeled, error_classes, on='id', how='left')

eval_data.groupby('higher_level_group').agg({'file_name' : 'count'})

def get_assigned(row): 
    if row['higher_level_group'] in ['Multiple', 'Not living', 'Corrupt']: 
        out = 'Not_plankton'
    else: 
        out = row['higher_level_group']
    return out

''' first see for how many of the images across all classes the coerced label is the same as the assigned label'''
eval_data['label_assigned'] = eval_data.apply(lambda row: get_assigned(row), axis=1)
eval_data['labels_match'] = eval_data.apply(lambda row: row['label_assigned'] == row['coerced_label'], axis=1)

sum(eval_data.labels_match) # wow this is astonishing, 388 of the 900 SME labels are right 512 are wrong

# -- 
# let summarize this by model error class 

# ''' binary classification error assesment '''

# e1a = eval_data.loc[eval_data['e_class'].isin(['other_to_not_plankton_bc', 'non_other_plankton_to_not_plankton_bc'])]
# e1a['error_is_false'] = e1a.apply(lambda row: row['coerced_label'] == 'Not_plankton', axis=1)
# sum(e1a.error_is_false) # 147/299 (49%)


''' binary classification error assesment for all error classes '''

e1a = eval_data.loc[eval_data['e_class'] == 'other_to_not_plankton_bc']
e1a['error_is_false'] = e1a.apply(lambda row: row['coerced_label'] == 'Not_plankton', axis=1)
sum(e1a.error_is_false) # 114/199 (57%)

e1b = eval_data.loc[eval_data['e_class'] == 'non_other_plankton_to_not_plankton_bc']
e1b['error_is_false'] = e1b.apply(lambda row: row['coerced_label'] == 'Not_plankton', axis=1)
sum(e1b.error_is_false) # 33/100 (33%)

e12 = eval_data.loc[eval_data['e_class'] == 'not_plankton_to_plankton_bc']
e12['error_is_false'] = e12.apply(lambda row: row['coerced_label'] != 'Not_plankton', axis=1)
sum(e12.error_is_false) #70/100 (70%)


# ''' multiclass classification error assesment '''

# error2_1a['e_class'] = 'plankton_to_other_mc'
# error2_1b['e_class'] = 'plankton_to_non_other_plankton_mc'
# error2_2a['e_class'] = 'other_to_plankton_mc'

# emc = eval_data.loc[eval_data['model'] == 'MC']
# emc['error_is_false'] = emc.apply(lambda row: row['coerced_label'] == row['pred_label_mc_string'], axis=1)
# sum(emc.error_is_false) # 172/500 (34%)


''' multiclass error assesment for specific error classes '''

e22a = emc.loc[emc['e_class'] == 'other_to_plankton_mc']
sum(e22a['error_is_false']) # 79 / 300

e21a = emc.loc[emc['e_class'] == 'plankton_to_other_mc']
sum(e21a['error_is_false']) # 81 / 100

e21b = emc.loc[emc['e_class'] == 'plankton_to_non_other_plankton_mc']
sum(e21b['error_is_false']) # 12 / 100


# -- 
# make confidence estimates for error accuracy 

from statsmodels import stats

''' multi-class confidence intervals '''
e22a_low, e22a_up = stats.proportion.proportion_confint(sum(e22a['error_is_false']), len(e22a), alpha=0.05, method='binom_test')
e21a_low, e21a_up = stats.proportion.proportion_confint(sum(e21a['error_is_false']), len(e21a), alpha=0.05, method='binom_test')
e21b_low, e21b_up = stats.proportion.proportion_confint(sum(e21b['error_is_false']), len(e21b), alpha=0.05, method='binom_test')

''' binary class confidence intervals '''
e1a_low, e1a_up = stats.proportion.proportion_confint(sum(e1a['error_is_false']), len(e1a), alpha=0.05, method='binom_test') #'other_to_not_plankton_bc'
e1b_low, e1b_up = stats.proportion.proportion_confint(sum(e1b['error_is_false']), len(e1b), alpha=0.05, method='binom_test')
e12_low, e12_up = stats.proportion.proportion_confint(sum(e12['error_is_false']), len(e12), alpha=0.05, method='binom_test')


# -- 
# translate to accuracy assesment --> this may be a kind of involved question, do simple emailable answer here 

n_e1a = len(error_classes.loc[error_classes['e_class'] == 'other_to_not_plankton_bc']) #4875 
_range = (n_e1a * e1a_low, n_e1a * e1a_up)

error_est = []
for frame in [e1a, e1b, e12, e22a, e21a, e21b]:
    desc = frame.iloc[0]['e_class']
    n_pop = len(error_classes.loc[error_classes['e_class'] == desc])
    low, up = stats.proportion.proportion_confint(sum(frame['error_is_false']), len(frame), alpha=0.05, method='binom_test')
    _range_low, _range_up = (n_pop * low, n_pop * up)
    out = {
        'class' : desc, 
        'count_error' : n_pop, 
        '95_conf_low_percent' : low, 
        '95_conf_up_percent' : up, 
        '95_conf_low_count' : _range_low, 
        '95_conf_up_count' : _range_up
    }
    error_est.append(out)

df_error = pd.DataFrame(error_est)

df_error.to_csv('./reference_data/error-class-confidence-estimates.csv', index=False)




