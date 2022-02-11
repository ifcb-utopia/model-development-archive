'''
    Analysis VI: 
        I: summarize model predictions and get sense of performance of binary classification
'''

# -- 
# dependancies 

from keras.models import model_from_json
from sklearn.metrics import classification_report
from model_function_library import * 
from data_io_function_library import * 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -- 
# io 

eval_data = pd.read_csv('./data/testing-data-preds-analysis-1.csv')


# -- 
# add flag for correct prediction 

def is_correct(row):
    if row['pred_label'] == row['true_label']: 
        out = 1
    else: 
        out = 0
    return out

eval_data['is_correct'] = eval_data.apply(lambda row: is_correct(row), axis=1) 


# -- 
# summarize accuracy of model predictions by high group assignment 

''' make reference file dataframe '''
fileRef = pd.read_csv('./reference_data/image-file-directory.csv')
fileRef = fileRef[['high_group', 'file_name', 'class_raw']]

''' join and aggregate to make summary '''
eval_data_comp = pd.merge(eval_data, fileRef, on='file_name', how='left')
eval_data_comp.reset_index(inplace=True)
NN_summary = eval_data_comp.groupby('high_group').agg({'is_correct' : 'sum', 
    'file_name' : 'count', 
    'label' : 'max'})

NN_summary.reset_index(inplace=True)
NN_summary['accuracy'] = NN_summary.apply(lambda row: row['is_correct'] / float(row['file_name']), axis=1)
performance_summary = NN_summary.sort_values(by =['label', 'file_name', 'accuracy'], ascending=[0, 0, 0] )
performance_summary.reset_index()
performance_summary.columns = ['higher_level_group', 'number_correct', 'total_examples', 'binary_label', 'classification_accuracy']

# -- 
# plot samples of results for aggregate classes and binary classes 

''' develop method to plot single plankton image '''

def get_image(row):
    input_path = row['path'] + row['label'] + '/' + row['file_name']
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


''' functionalize plotting process ''' 

def make_plot(_class):
    test_sample = eval_data_comp.loc[eval_data_comp['high_group'] == _class]
    shuffled = test_sample.sample(frac=1)
    correct = shuffled.loc[test_sample['is_correct'] == 1].sample(n=4)
    incorrect = shuffled.loc[test_sample['is_correct'] == 0].sample(n=4)
    class_frame = pd.concat([correct, incorrect])
    class_frame['is_correct_string'] = class_frame.apply(lambda row: string_is_correct(row), axis=1)
    class_frame = pd.merge(class_frame, performance_summary[['high_group', 'accuracy']], on='high_group', how='left')
    cases = class_frame.class_raw.values.tolist()
    row = class_frame.iloc[0]
    fig1, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    fig1.suptitle('Examples of correct (top) & incorrect (bottom) prediction for ' + \
        _class + ' image class: ' + str(round(row['accuracy'] * 100, 2)) + ' percent accuracy')
    axs = trim_axs(axs, len(cases))
    t = zip(axs, cases)
    for i in range(8):
        ax, case = t[i]
        row = class_frame.iloc[i]
        img = get_image(row)
        ax.imshow(img)  
        ax.set_title(str(case))
    plt.savefig('./plots/binary-class-plot-' + _class + '.png')
    plt.close()


''' run for specific set of image classes '''

class_sub = performance_summary.loc[performance_summary['file_name'] >=100]
for n in range(len(class_sub)): 
    _class = class_sub.iloc[n]['high_group']
    make_plot(_class)


