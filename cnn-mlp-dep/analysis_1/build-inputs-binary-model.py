'''
    Analysis script II: 
        I: make initial non-discretionary exclusions to data (done)
        II: prepare samples for train and test (done)
        III: structure files directory of images ()
        IV: make vector data inputs (done)
'''

# -- 
# dependancies 

from data_io_function_library import *


# -- 
# io 

data = pd.read_csv('./reference_data/image-file-directory.csv')


# -- 
# make initial upfront exclusions to raw data 

image_subset_a = data.loc[data['missing_meta_data'] == False]
print('have excluded ' + str(len(data) - len(image_subset_a)) + ' images with no meta data: ' + str(len(image_subset_a)) + ' remaining')

''' exclude images below 7.1um threshold '''

data_subset = image_subset_a.loc[image_subset_a['ESDA_exclude'] == False]
print('have excluded ' + str(len(image_subset_a) - len(data_subset)) + ' images bellow size threshold: ' + str(len(data_subset)) + ' remaining')


# -- 
# prepare samples for model training 

test_sample = data_subset.loc[data_subset['set'] == 'test'] # 215,357
train_raw = data_subset.loc[data_subset['set'] == 'train'] # 1,219,656


# -- 
# look at distrbution of test / train set without upsampling / downsampling 

test_agg = test_sample.groupby('high_group').agg({'binary_label' : 'max', 'file_name' : 'count', 'class_raw' : pd.Series.nunique})
train_agg = train_raw.groupby('high_group').agg({'binary_label' : 'max', 'file_name' : 'count', 'class_raw' : pd.Series.nunique})


''' build valid / unidentifiable plankton vs not plankton / non living sample using all the available data '''

plankton = train_raw.loc[train_raw['binary_label'] == 'plankton']
not_plankton = train_raw[train_raw['binary_label'] == 'not_plankton']


''' sample planktonic data '''

plank_summary = plankton.groupby('high_group').agg({'file_name' : 'count',
    'class_raw' : pd.Series.nunique})
plank_summary.reset_index(inplace=True)
significant_groups = plank_summary.high_group.loc[plank_summary['file_name'] >= 1000]
plankton_sub = plankton.loc[plankton['high_group'].isin(significant_groups)]
raw_class_summary = plankton_sub.groupby('class_raw').agg({'file_name' : 'count', 
    'binary_label' : 'max', 
    'high_group' : 'max'})
raw_class_summary.reset_index(inplace=True) # 63

sig_raw_classes = raw_class_summary.loc[raw_class_summary['file_name'] >= 500]

multiple_sig_classes = sig_raw_classes.groupby('high_group').agg({'class_raw' : 'count'})
multiple_sig_classes.reset_index(inplace=True) 
groups_sub_sample = multiple_sig_classes.loc[multiple_sig_classes['class_raw']>1].high_group
groups_no_sub_sample = multiple_sig_classes.loc[multiple_sig_classes['class_raw']==1].high_group

group_frames = [] 
for i in groups_no_sub_sample: 
    subset = plankton.loc[plankton['high_group'] == i]
    if plank_summary.loc[plank_summary['high_group'] == i]['file_name'].values[0] < 10000: 
        subset_out = plankton.loc[plankton['high_group'] == i]
        upsample_list = [subset_out for i in range(10000 / len(subset_out))]
        group_frames.append(pd.concat(upsample_list)) 
    else: 
        subset_out = plankton.loc[plankton['high_group'] == i]
        group_frames.append(subset_out)


for i in groups_sub_sample: 
    sub_classes = sig_raw_classes.loc[sig_raw_classes['high_group'] == i].class_raw
    class_samp = []
    for c in sub_classes: 
        samp = plankton.loc[plankton['class_raw'] == c]
        class_samp.append(samp)
        # 
    class_sample = pd.concat(class_samp)
    if len(class_sample) < 10000: 
        # upsample_list = [class_sample for i in range(10000 / len(class_sample))]
        nUp = 10000 - len(class_sample)
        upsamp = class_sample.sample(n=nUp)
        group_frames.append(pd.concat([class_sample, upsamp]))
    else: 
        group_frames.append(class_sample)


plankton = pd.concat(group_frames)

''' check resulting sample distribution '''

plank_agg = plankton.groupby('high_group').agg({'binary_label' : 'max', 'file_name' : 'count', 'class_raw' : pd.Series.nunique})
plank_agg.reset_index(inplace = True)


''' get a sense of the duplication of images in these class sub samples '''

def get_unique(row): 
    frame = plankton.loc[plankton['high_group'] == row['high_group']]
    vals = frame.file_name.values.tolist()
    n_unique = len(list(set(vals)))
    return n_unique

plank_agg['unique_examples'] = plank_agg.apply(lambda row: get_unique(row), axis=1)
plank_agg['duplicate_examples'] = plank_agg.apply(lambda row: row['file_name'] - row['unique_examples'], axis=1)

''' sample non planktonic data '''

feces = not_plankton.loc[not_plankton['class_raw'] == 'Feces']
feces_up = pd.concat([feces for i in range(4)])

multiple = not_plankton.loc[not_plankton['class_raw'] == 'Multiple']
multiple_up = pd.concat([multiple for i in range(4)])

NP_a = not_plankton.loc[~not_plankton['class_raw'].isin(['Feces', 'Multiple'])]
NPa_up = pd.concat([NP_a for i in range(3)])

np = pd.concat([feces_up, multiple_up, NPa_up])


''' combine into train sample frame: rescue some examples from train raw that we missed in sampling process '''

train_sample = pd.concat([plankton, np]).sample(n = len(np) + len(plankton))

test_files = test_sample.file_name.values.tolist()
train_files = train_sample.file_name.values.tolist()
all_files = list(set(train_files + test_files))

missed_files = train_raw.loc[~train_raw['file_name'].isin(train_files)]
train_sample = pd.concat([train_sample, missed_files])

test_files = test_sample.file_name.values.tolist()
train_files = train_sample.file_name.values.tolist()
all_files = list(set(train_files + test_files))


# -- 
# produce vector data inputs for training and testing samples 

sample_data = pd.concat([train_sample, test_sample])
sample_data['label'] = sample_data['binary_label']


vector_data = sample_data[[u'Area', u'Biovolume', u'ConvexArea', u'ConvexPerimeter',
       u'FeretDiameter', u'MajorAxisLength', u'MinorAxisLength', u'Perimeter',
       u'ESDA', u'ESDV', u'PA', u'ScatInt', u'FluoInt', u'ScatPeak',
       u'FluoPeak', u'NumberOfROIinTrigger', u'label',
       u'file_name']]

''' write out vector data for export to remote machine '''

vector_data.to_csv('./data/vector-data-analysis-1.csv', index=False)


# -- 
# assign files in train and test samples to directories for zipping and scp

if not os.path.exists('/Users/culhane/Desktop/NAAMES/train/'): 
    os.mkdir('/Users/culhane/Desktop/NAAMES/train/')

if not os.path.exists('/Users/culhane/Desktop/NAAMES/test/'): 
    print('making new directory for testing images')
    os.mkdir('/Users/culhane/Desktop/NAAMES/test/')


''' rework image transportation function '''

train_images = list(set(train_sample.file_name_x.values.tolist()))
test_images = list(set(test_sample.file_name_x.values.tolist()))

def move_files(file_list, _set):
    plankton_files_home = '/Users/culhane/Desktop/NAAMES/'
    path = plankton_files_home + _set + '/'
    for i in range(len(file_list)): 
        f = file_list[i]
        to_path = path + f
        from_path = plankton_files_home + f
        os.rename(from_path, to_path)


move_files(test_images, 'test')
move_files(train_images, 'train')

len(os.listdir('/Users/culhane/Desktop/NAAMES/test/')) == len(test_images)
len(os.listdir('/Users/culhane/Desktop/NAAMES/train/')) == len(train_images)
