'''
    Analysis VII: 
        I: Build vector data for species level classification network
'''

# -- 
# dependancies

from model_function_library import * 
from data_io_function_library import * 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pprint import pprint


# -- 
# io 

data = pd.read_csv('./reference_data/image-file-directory.csv')
data_grouped = data.groupby('high_group').agg({'binary_label' : 'max', 'file_name' : 'count'})
data_grouped.reset_index()


# -- 
# make train data 

def assign_species_label(row): 
    if row['high_group'] in ['Chryso', 'Cyanobacteria', 'Rhizaria', 'Zoo']: 
        lab = 'Other'
    elif row['high_group'] in ['Artefact', 'Corrupt', 'Multiple', 'Not living']: 
        lab = 'Not_plankton'
    else: 
        lab = row['high_group']
    return lab

data['cat_label'] = data.apply(lambda row: assign_species_label(row), axis=1)


# -- 
# load vector data, join and replace label 

data_ref = data[['file_name', 'cat_label']]
vector_data = pd.read_csv('./data/vector-data-analysis-1.csv')
vector_data = pd.merge(vector_data, data_ref, on='file_name', how='left')

''' reassign label to cat_label to allow same code to be used '''

vector_data['label'] = vector_data['cat_label']
vector_data = vector_data.drop('cat_label', axis=1)

train_p1 = vector_data.loc[vector_data['set'] == 'train'] #1605689
test_p1 = vector_data.loc[vector_data['set'] == 'test'] #215357

train_images = os.listdir('/Users/culhane/Desktop/NAAMES/train/') #1219656
test_images = os.listdir('/Users/culhane/Desktop/NAAMES/test/') #215357

# -- 
# get labels as set and perform one hot encoding 

lb = LabelBinarizer()
labels = set(vector_data.label)
lb.fit(list(labels))

''' build reference dictionary of label one hot encoding '''

encoding = {} 
for l in labels: 
    encoding[l] = lb.transform([l])

pprint(encoding)

''' 
** One hot encoded plankton classes for species level network **

{'Chloro': array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'Cilliate': array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'Crypto': array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]),
 'Diatom': array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]),
 'Dictyo': array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
 'Dinoflagellate': array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]),
 'Eugleno': array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]),
 'Not_plankton': array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]),
 'Other': array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]),
 'Prymnesio': array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])}

'''

# -- 
# make vector data inputs for new 

def is_correct(row):
    if row['pred_label'] == row['true_label']: 
        out = 1
    else: 
        out = 0
    return out


''' load predictions from binary classification / subset testing data '''

test_preds = pd.read_csv('./data/testing-data-preds-analysis-1.csv')
test_preds['file_name'] = test_preds['file_name_x']
test_preds = test_preds.drop('file_name_x', axis=1)
test_preds['is_correct'] = test_preds.apply(lambda row: is_correct(row), axis=1)


''' produce testing data vector dataset '''

test_SLC_files = test_preds.loc[test_preds['pred_label'] == 1].file_name
test_vector = vector_data.loc[vector_data['file_name'].isin(test_SLC_files)] #179175
train_vector = vector_data.loc[vector_data['set']=='train'] #1605689


''' make sure vector data already has upsampling '''

len(list(set(vector_data.file_name_x))) < len(vector_data.file_name_x)
len(list(set(vector_data.file_name_x))) != len(vector_data.file_name_x)


''' do basic version of sampling; just subset 'Other' and downsampled Not_plakton in proportion '''

train_other = train_vector.loc[train_vector['label'] == 'Other'].sample(n=160000)
train_notplank = train_vector.loc[train_vector['label'] == 'Not_plankton'].sample(n=50000)
train_plank = train_vector.loc[~train_vector['label'].isin(['Other', 'Not_plankton'])]
train_vector_data = pd.concat([train_other, train_notplank, train_plank])

''' combine train and test vector data for multiclass network baseline & write out for io'''

vector_data_multi = pd.concat([train_vector_data, test_vector])
vector_data_multi.columns
vector_data_multi.to_csv('./data/vector-data-analysis-1-p2.csv', index=False)
