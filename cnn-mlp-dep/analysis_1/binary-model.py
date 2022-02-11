'''
    Analysis script IV: 
        I: implement multi input binary classification network 
        https://keras.io/getting-started/sequential-model-guide/
'''

# --
# dependancies 

''' own modules and functionality '''

from model_function_library import * 
from data_io_function_library import * 
from keras.callbacks import EarlyStopping, ModelCheckpoint


# -- 
# io 

vector_path = './data/vector-data-analysis-1.csv'
vector_data = pd.read_csv(vector_path)

train = vector_data.loc[vector_data['set'] == 'train'].drop('set', axis=1)
test = vector_data.loc[vector_data['set'] == 'test'].drop('set', axis=1)



# -- 
# use modules to define the multi input generators 

''' parameters for generators '''

train_path = './data/train/'
test_path = './data/test/'
lb = LabelBinarizer()
batch_size = 50


''' instantiate generators '''

trainGen = multi_stream_generator_final(train, train_path, batch_size, lb)
testGen = multi_stream_generator_final(test, test_path, batch_size, lb)


# -- 
# define models & specify multi-input interaction 

mlp = create_mlp_model1b(16, regress=False)
cnn = create_cnn_model1b(128, 128, 3, regress=False)
combinedInput = concatenate([mlp.output, cnn.output])
x = Dense(1000, activation="relu")(combinedInput)
x = Dropout(rate=0.1)(x)
x = Dense(512, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=[cnn.input, mlp.input], outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# -- 
# add callback for early stopping of training iterations 

callbacks = [ModelCheckpoint(filepath='best_model_analysis_1.h5', monitor='val_loss', save_best_only=True)]


# -- 
# train model with generator flow of multi input threads 

totalTrain = len(train)
totalVal = len(test)

print("[INFO] training simple network...")
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // batch_size,
    validation_data=testGen,
    validation_steps=totalVal // batch_size,
    epochs=20,
    callbacks=callbacks)


# -- 
# save model & load and predict (hopefully this worked !!!)

''' save model and weights '''
model_json = model.to_json()
with open("analysis-1.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("analysis-1.h5")
print("Saved model to disk")

