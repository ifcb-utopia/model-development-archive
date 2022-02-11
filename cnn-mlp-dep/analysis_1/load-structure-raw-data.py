'''
    Build reference file for images with data availability flags: 
        I: join to taxonomic reference
        II: coerce missing higher level groups 
        III: add coerced higher level flag 
        III: join to metadata file 
        IV: add missing meta data flag
        V: add ESDA exclusion flag 
'''

# -- 
# dependancies 

import pandas as pd 
import os
import re
import matplotlib.pyplot as plt
from pprint import pprint

# -- 
# io 

''' taxonomic reference file '''
df_tax = pd.read_csv('./reference_data/taxonomic_grouping_v3.csv')

''' metadata file '''
df_meta = pd.read_csv('./reference_data/master.csv')

''' image file name directory: analysis 1 data has 1998900 examples '''
image_path = '/Users/culhane/Desktop/NAAMES/'
_dirs = [i for i in os.listdir(image_path) if i in ['train', 'test', 'validation']]
_files = [i for i in os.listdir(image_path) if i not in ['train', 'test', 'validation']]

for d in _dirs: 
    _f = os.listdir(image_path + d)
    _files.extend(_f)

df_files = pd.DataFrame(_files)
df_files.columns = ['file_name']



# -- 
# join to taxonomic reference 

def get_name(row): 
    fn = row['file_name']
    subs = re.compile("\_(.*?).png")
    try: 
        out = re.findall(subs, fn)[0].replace('_', '').replace('.png', '')
    except: 
        out = 'invalid_file_name'
    return out

def link_names(row): 
    cr = row['class_raw'].lower()
    for i in range(len(high_groups)): 
        classes = [x.lower() for x in high_groups.iloc[i]['category_prettified']]
        if cr in classes: 
            out = high_groups.iloc[i]['category_grouped']
            break
        else: 
            out = None
    return out


# -- 
# apply and structure data & join taxonomy data to aggregate of files

high_groups = df_tax.groupby('category_grouped').agg({'category_prettified' : lambda x: list(set(x))})
high_groups.reset_index(inplace=True)

df_files['class_raw'] = df_files.apply(lambda row: get_name(row), axis=1)
classes_grouped = df_files.groupby('class_raw').agg({'file_name' : 'count'})
classes_grouped.reset_index(inplace=True)
classes_grouped['high_group'] = classes_grouped.apply(lambda row: link_names(row), axis=1)

df_files = pd.merge(df_files, classes_grouped, on = 'class_raw', how='left')


# -- 
# manually coerce high group assignment with reference dictionary for missed high groups 

df_files.high_group.unique()
df_files['missing_high_group'] = df_files.apply(lambda row: row['high_group'] == None, axis=1)

no_group_images = df_files.loc[df_files['missing_high_group'] == True]
no_group_summary = no_group_images.groupby('class_raw').agg({'file_name_x' : 'count'})
no_group_summary.reset_index(inplace=True)

lookup_no_class = {'Bacillariophyta-centric' : 'Diatom',
                    'Bacillariophyta-centric-chain' : 'Diatom',
                    'Bacillariophyta-pennate': 'Diatom',
                    'Bacillariophyta-pennate-chain': 'Diatom',
                    'Bad-focus': 'Corrupt',
                    'Chaetoceros-spore': 'Other',
                    'Cladopyxis-brachiolata': 'Other',
                    'Degraded-': 'Corrupt',
                    'Detritus-fiber': 'Not living',
                    'Dividing-cells': 'Other',
                    'Guinardia-delicatula': 'Other',
                    'LeptoEPI': 'Other',
                    'Other-part': 'Other',
                    'Other-to-check': 'Other',
                    'Plastic-fiber': 'Not living',
                    'Plastic-fragment': 'Not living',
                    'Plastic-other': 'Not living',
                    'Scyphosphaera-apsteinii': 'Other',
                    'T1-CoolCircle': 'Other',
                    'T12-Unknown': 'Other',
                    'T15-None': 'Other',
                    'T4-None': 'Other',
                    'T8-None': 'Other',
                    'Tintinnidiidae-empty': 'Other',
                    'Unicellular-cyst': 'Other'}

# -- 
# fill all missing values using above reference dictionary 

def fill_na(row): 
    if row['missing_high_group'] == True:
        out = lookup_no_class[row['class_raw']]
    else: 
        out = row['high_group']
    return out


df_files = df_files.drop(['file_name_y'], axis=1)
df_files.columns = ['file_name', 'class_raw', 'high_group', 'missing_high_group']
df_files['high_group'] = df_files.apply(lambda row: fill_na(row), axis=1)

df_group = df_files.groupby('high_group').agg({'file_name' : 'count'})
df_group = df_group.reset_index()


# -- 
# join to metadata 

''' separate id from file name for join to metadata file '''

def get_id(row): 
    fn = row['file_name']
    return re.match('(.*?)\_', fn).groups()[0]

df_files['id'] = df_files.apply(lambda row: get_id(row), axis=1)
df_files = pd.merge(df_files, df_meta, on='id', how='left')
df_files['missing_meta_data'] = df_files.apply(lambda row: pd.isnull(row['Area']), axis=1)


# -- 
# add the ESDA exclusion flag 

df_files['ESDA_exclude'] = df_files.apply(lambda row: row['ESDA'] <= 7.1, axis=1)
sum(df_files.ESDA_exclude)


# -- 
# write out new set of files 

df_files.to_csv('./reference_data/image-file-directory.csv', index=False)






