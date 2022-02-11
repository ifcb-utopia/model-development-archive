
'''
    Analysis script IX: 
        I: Evaluate model predictions for multi-class model 

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

''' model import: weights and layers '''

json_file = open('analysis-1-p2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("analysis-1-p2.h5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

''' vector data from training and testing '''

vector_path = './data/vector-data-analysis-1-p2.csv'
vector_data = pd.read_csv(vector_path)

test = vector_data.loc[vector_data['set'] == 'test'].drop('set', axis=1)
del vector_data

# -- 
# use modules to define the multi input generators 

''' parameters for generators '''

test_path = './data/test/'
lb = LabelBinarizer()
labels = set(vector_data.label)
lb.fit(list(labels))
batch_size = 50


''' instantiate generators '''

testGen = multi_stream_generator_final(test, test_path, batch_size, lb)

# -- 
# generate combined dataset from testing and validation and asses model performance 

''' data manipulation functions  '''

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

def is_correct(row):
    if row['pred_label'] == row['true_label']: 
        out = 1
    else: 
        out = 0
    return out


''' sturcture for prediction (using testing and validation data / joining to full frame '''

test['path'] = test_path


''' split data into 4 sub frames and generate predictions '''

test_split = np.array_split(test, 4)
test_preds = []
n = 1
for df in test_split: 
    print('getting labels and vector set')
    labels = df['label'].values
    labels = lb.fit_transform(labels)
    v_dat = df.drop(['file_name_x', 'label', 'path'], axis=1).values
    print('loading and transforming image data')
    image_data = []
    for i in range(len(df)): 
        row = df.iloc[i]
        input_path = row['path'] + row['file_name_x']
        image_data.append(preprocess_input(cv2.imread(input_path)))
        #
    image_data = np.array(image_data)
    test_input = [ image_data, v_dat ]
    test_labels = labels
    print('generating predictions for testing subset')
    predictions = loaded_model.predict(test_input)
    pred_indices = [np.argmax(i) for i in predictions]
    df['pred_label'] = pred_indices
    df['true_label'] = [np.argmax(i) for i in test_labels]
    df['is_correct'] = df.apply(lambda row: is_correct(row), axis=1)
    test_preds.append(df)
    print('completed evaluation of ' + str(n) + ' subset of testing dataset')
    n +=1 
    del image_data


test_eval = pd.concat(test_preds)
len(test_eval) == len(test)

# -- 
# write out data and send to local machine 

''' write out csv of testing data and predictions '''

test_eval.to_csv('./testing-data-preds-analysis-1-p2.csv', index=False)

''' scp testing data from remote ec2 to local machine '''

scp -i plankton-net.pem ec2-user@ec2-13-58-33-153.us-east-2.compute.amazonaws.com:~/plankton/testing-data-preds-analysis-1-p2.csv .

''' scp model checkpoints '''

scp -i plankton-net.pem ec2-user@ec2-13-58-33-153.us-east-2.compute.amazonaws.com:~/plankton/best_model_analysis_1_p2.h5 .
scp -i plankton-net.pem ec2-user@ec2-13-58-33-153.us-east-2.compute.amazonaws.com:~/plankton/analysis-1-p2.json .
scp -i plankton-net.pem ec2-user@ec2-13-58-33-153.us-east-2.compute.amazonaws.com:~/plankton/analysis-1-p2.h5 .


