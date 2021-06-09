## built in libraries
from tensorflow import keras
import scipy
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import imblearn as imbl
## user's libraries 
from data_io import adjust_batches_egms, adjust_batches, DataGenerator, data, load_data_generator, data_generator, get_f_names, fnames_split_train_val_test, balance_oversampling_files, read_data_for_ros
from train_graph import STD_classifier
from evaluate_model import plots, eval_and_store_gen
from math import ceil
import numpy as np 

## Seems to be useful for incremental memory allocation, otherwise TF consumes all memory of the GPU instantaniously
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

## metaparameters
parameters = {
        "is_it_CNN2D": False, # True, # 
        "architecture": "mlr", # "lenet", # 
        "n_folds": 5,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 50,
        "augment" : True, # False,  #   ## oversampling"
        "monitor" : "val_auc",
        "patience" : 10,
        "mode" : "auto",
        "num_classes" : 2, 
        "steps_per_epoch" : None, #10,
        "validation_steps" : None, #10,
        "dim": (2500,12),
        "shuffle": False, # True,
    }
# add 4th dim if 2D CNN
if parameters["is_it_CNN2D"]:
    input_shape = (2500, 12, 1)
else:
    input_shape = (2500, 12) # list(x_train.shape[1:])

## directories
model_dir = os.path.join('DL_STD_classif','classif','debug_'+parameters["architecture"]+'_generator')
data_dir = os.path.join('DL_STD_classif',)

if parameters["augment"]:
    model_dir += '_ros'    
if not os.path.exists(model_dir): # Create target Directory if doesn't exist
    os.mkdir(model_dir)

## split files_names 
f_names = get_f_names()
train_files, val_files, test_files = fnames_split_train_val_test(f_names)

# oversampling(train file_names)
if parameters["augment"]:
    print('ros training dataset')
    train_files, y_train = balance_oversampling_files(train_files)
    nb_train_aug = len(train_files)
    p = np.random.permutation(nb_train_aug)
    train_files = [train_files[i] for i in p]
    y_train = y_train[p]

train_files, y_train, x_train = adjust_batches_egms(train_files, parameters["batch_size"])
val_files, y_val, x_val = adjust_batches_egms(val_files, parameters["batch_size"])
test_files, y_test, x_test = adjust_batches_egms(test_files, parameters["batch_size"])

# parameters["steps_per_epoch"] = ceil(len(train_files) / parameters["batch_size"])
# parameters["validation_steps"] = ceil( len(val_files) / parameters["batch_size"])

partition = {
        'train': train_files, 
        'validation': val_files,
        'test' : test_files,  
    }
    
## Generators
train_generator = DataGenerator(partition['train'], parameters)
val_generator = DataGenerator(partition['validation'], parameters)
test_generator = DataGenerator(partition['test'], parameters)

# Built & run graph
std_classifier = STD_classifier(parameters,input_shape, model_dir)
architecture = std_classifier.get_architecture()
model, history, elapsed = std_classifier.trainer_gen(architecture,std_classifier.call_functions(), train_generator, val_generator) # (x_val, y_val)) #, wandb)
# model, history, elapsed = std_classifier.trainer(architecture,std_classifier.call_functions(), x_train, y_train, x_val, y_val) #, wandb)
print("trained")

model.save(os.path.join(model_dir,'model.h5'))

" works only if its input data is an interator: train_generator, val_generator, test_generator"
history = history.history
eval_and_store_gen(model, model_dir, parameters["augment"], history, test_generator, elapsed)
plots(model_dir, history)

print("evaluated")

""" Debug model """
from sklearn.metrics import confusion_matrix

y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
y_test_pred = model.predict(x_test)

cm_train = confusion_matrix(y_train,y_train_pred, normalize = 'true')
cm_val = confusion_matrix(y_val,y_val_pred, normalize = 'true')
cm_test = confusion_matrix(y_test,y_test_pred, normalize = 'true')

##########################################################################################################
##########################################################################################################

""" Classical train - Generator  """
from def_models import mlr, lenet
from tensorflow.keras.optimizers import Adam

model1 = mlr(2, input_shape)

metrics_to_compute = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'), # CategoricalAccuracy(name='accuracy'), # 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        ]
model1.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), #CategoricalCrossentropy(), 'categorical_crossentropy', # tf.keras.losses.CategoricalCrossentropy(), #
    optimizer=Adam(parameters["learning_rate"]),
    metrics=metrics_to_compute)

csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'training_classical.csv'))
early = tf.keras.callbacks.EarlyStopping(monitor=parameters["monitor"], patience=parameters["patience"],mode=parameters["mode"])
calls = [csv_logger, early]

history1 = model1.fit(x_train, y_train,
          batch_size=parameters["batch_size"],
          epochs=parameters["epochs"],
          verbose=1,
          validation_data=(x_val,y_val), callbacks=calls)

history1 = history1.history

training_acc = history1['accuracy']
training_loss = history1['loss']
val_acc = history1['val_accuracy']
val_loss = history1['val_loss']
training_AUC = history1['auc']
val_AUC = history1['val_auc']

score_MLR = model1.evaluate(x_test, y_test, verbose=1)
# score_MLR list corresponds to  model.metrics_names = ['loss', 'auc', 'acc']

y_train_pred1 = model1.predict(x_train)
y_val_pred1 = model1.predict(x_val)
y_test_pred1 = model1.predict(x_test)

cm_train1 = confusion_matrix(y_train,y_train_pred1, normalize = 'true')
cm_val1 = confusion_matrix(y_val,y_val_pred1, normalize = 'true')
cm_test1 = confusion_matrix(y_test,y_test_pred1, normalize = 'true')

with open(os.path.join(model_dir,'model1.txt'),'a') as f_conf:
    f_conf.write(str(model1.metrics_names) +'\n')
    f_conf.write('test score = '+ str(score_MLR)+'\n')
    f_conf.write('cm_train1 = '+ str(cm_train1)+'\n')
    f_conf.write('cm_val1 = '+ str(cm_val1)+'\n')
    f_conf.write('cm_test1 = '+ str(cm_test1)+'\n')

##########################################################################################################
##########################################################################################################
""" Adam + binary loss - Generator"""
# convert class vectors to binary class matrices

## LR model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

input_shape = list(x_train.shape[1:])

inputs = Input(shape=input_shape)
feature = Flatten()(inputs)
feature = Dense(1, activation='softmax')(feature)
model0_deb2 = Model(inputs=inputs, outputs=feature)

## Compile

monitor = 'val_auc'
patience = 10
mode = 'auto'
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'adam_binary_training_deb.csv'))
early = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,mode=mode)
calls = [csv_logger, early]

loss = tf.keras.losses.BinaryCrossentropy() # 'categorical_crossentropy' # 'mse'# 
optim = keras.optimizers.Adam()
#metrics_to_compute = [tf.keras.metrics.AUC(), 'acc']
metrics_to_compute = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'), # CategoricalAccuracy(name='accuracy'), # 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        ]
model0_deb2.compile(loss=loss,
              optimizer=Adam(parameters["learning_rate"]), # optim,
              metrics=metrics_to_compute)
          
## Training
history_deb2 = model0_deb2.fit(x_train, y_train,
          batch_size=parameters["batch_size"],
          epochs=parameters["epochs"],
          verbose=1,
          validation_data=(x_val,y_val), callbacks=calls)

training_acc = history_deb2.history['accuracy']
training_loss = history_deb2.history['loss']
val_acc = history_deb2.history['val_accuracy']
val_loss = history_deb2.history['val_loss']
training_AUC = history_deb2.history['auc']
val_AUC = history_deb2.history['val_auc']

score_MLR0_deb1 = model0_deb2.evaluate(x_test, y_test, verbose=1)
# score_MLR list corresponds to  model.metrics_names = ['loss', 'auc', 'acc']

y_train_pred = model0_deb2.predict(x_train)
y_val_pred = model0_deb2.predict(x_val)
y_test_pred = model0_deb2.predict(x_test)


cm_train0_deb1 = confusion_matrix(y_train,y_train_pred, normalize = 'true')
cm_val0_deb1 = confusion_matrix(y_val,y_val_pred, normalize = 'true')
cm_test0_deb1 = confusion_matrix(y_test,y_test_pred, normalize = 'true')

with open(os.path.join(model_dir,'model0_deb2.txt'),'a') as f_conf:
    f_conf.write(str(model0_deb2.metrics_names) +'\n')
    f_conf.write('test score = '+ str(score_MLR0_deb1)+'\n')
    f_conf.write('cm_train1 = '+ str(cm_train0_deb1)+'\n')
    f_conf.write('cm_val1 = '+ str(cm_val0_deb1)+'\n')
    f_conf.write('cm_test1 = '+ str(cm_test0_deb1)+'\n')

##########################################################################################################
##########################################################################################################

""" Classical training + Classical parameters  """
# convert class vectors to binary class matrices
y_train_1_column = y_train            # keep a copy of  y_test  before 1-hot transofmation -> useful for CM calculation
y_val_1_column = y_val            # keep a copy of  y_test  before 1-hot transofmation -> useful for CM calculation
y_test_1_column = y_test            # keep a copy of  y_test  before 1-hot transofmation -> useful for CM calculation

y_train = tf.keras.utils.to_categorical(y_train, 2)
y_val = tf.keras.utils.to_categorical(y_val, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

## LR model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

input_shape = list(x_train.shape[1:])

inputs = Input(shape=input_shape)
feature = Flatten()(inputs)
feature = Dense(2, activation='softmax')(feature)
model0 = Model(inputs=inputs, outputs=feature)

## Compile

monitor = 'val_auc'
patience = 10
mode = 'auto'
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'old_training.csv'))
early = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,mode=mode)
calls = [csv_logger, early]


loss = 'categorical_crossentropy' # 'mse'# 
optim = keras.optimizers.Adam()
metrics_to_compute = [tf.keras.metrics.AUC(), 'acc']
model0.compile(loss=loss,
              optimizer=optim,
              metrics=metrics_to_compute)
          
## Training
history_MLR = model0.fit(x_train, y_train,
          batch_size=parameters["batch_size"],
          epochs=parameters["epochs"],
          verbose=1,
          validation_data=(x_val,y_val), callbacks=calls)

training_acc = history_MLR.history['acc']
training_loss = history_MLR.history['loss']
val_acc = history_MLR.history['val_acc']
val_loss = history_MLR.history['val_loss']
training_AUC = history_MLR.history['auc']
val_AUC = history_MLR.history['val_auc']

score_MLR0 = model0.evaluate(x_test, y_test, verbose=1)
# score_MLR list corresponds to  model.metrics_names = ['loss', 'auc', 'acc']

y_train_pred = model0.predict(x_train)
y_val_pred = model0.predict(x_val)
y_test_pred = model0.predict(x_test)

y_train_pred =np.argmax(y_train_pred, axis = 1)
y_val_pred = np.argmax(y_val_pred, axis = 1)
y_test_pred = np.argmax(y_test_pred, axis = 1)

cm_train0 = confusion_matrix(y_train_1_column,y_train_pred, normalize = 'true')
cm_val0 = confusion_matrix(y_val_1_column,y_val_pred, normalize = 'true')
cm_test0 = confusion_matrix(y_test_1_column,y_test_pred, normalize = 'true')

with open(os.path.join(model_dir,'model0.txt'),'a') as f_conf:
    f_conf.write(str(model0.metrics_names) +'\n')
    f_conf.write('test score = '+ str(score_MLR0)+'\n')
    f_conf.write('cm_train1 = '+ str(cm_train0)+'\n')
    f_conf.write('cm_val1 = '+ str(cm_val0)+'\n')
    f_conf.write('cm_test1 = '+ str(cm_test0)+'\n')

##########################################################################################################
##########################################################################################################

""" DEBUG Classical training + Classical parameters  """
" add metrics to compute"
# convert class vectors to binary class matrices

## LR model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

input_shape = list(x_train.shape[1:])

inputs = Input(shape=input_shape)
feature = Flatten()(inputs)
feature = Dense(2, activation='softmax')(feature)
model0_deb = Model(inputs=inputs, outputs=feature)

## Compile

monitor = 'val_auc'
patience = 10
mode = 'auto'
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'old_training_deb.csv'))
early = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,mode=mode)
calls = [csv_logger, early]

loss = 'categorical_crossentropy' # 'mse'# 
optim = keras.optimizers.Adam()
#metrics_to_compute = [tf.keras.metrics.AUC(), 'acc']
metrics_to_compute = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'), # CategoricalAccuracy(name='accuracy'), # 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        ]
model0_deb.compile(loss=loss,
              optimizer=optim,
              metrics=metrics_to_compute)
          
## Training
history_deb = model0_deb.fit(x_train, y_train,
          batch_size=parameters["batch_size"],
          epochs=parameters["epochs"],
          verbose=1,
          validation_data=(x_val,y_val), callbacks=calls)

training_acc = history_deb.history['accuracy']
training_loss = history_deb.history['loss']
val_acc = history_deb.history['val_accuracy']
val_loss = history_deb.history['val_loss']
training_AUC = history_deb.history['auc']
val_AUC = history_deb.history['val_auc']

score_MLR0_deb = model0_deb.evaluate(x_test, y_test, verbose=1)
# score_MLR list corresponds to  model.metrics_names = ['loss', 'auc', 'acc']

y_train_pred = model0_deb.predict(x_train)
y_val_pred = model0_deb.predict(x_val)
y_test_pred = model0_deb.predict(x_test)

y_train_pred =np.argmax(y_train_pred, axis = 1)
y_val_pred = np.argmax(y_val_pred, axis = 1)
y_test_pred = np.argmax(y_test_pred, axis = 1)

cm_train0_deb = confusion_matrix(y_train_1_column,y_train_pred, normalize = 'true')
cm_val0_deb = confusion_matrix(y_val_1_column,y_val_pred, normalize = 'true')
cm_test0_deb = confusion_matrix(y_test_1_column,y_test_pred, normalize = 'true')

with open(os.path.join(model_dir,'model0_deb.txt'),'a') as f_conf:
    f_conf.write(str(model0_deb.metrics_names) +'\n')
    f_conf.write('test score = '+ str(score_MLR0_deb)+'\n')
    f_conf.write('cm_train1 = '+ str(cm_train0_deb)+'\n')
    f_conf.write('cm_val1 = '+ str(cm_val0_deb)+'\n')
    f_conf.write('cm_test1 = '+ str(cm_test0_deb)+'\n')

##########################################################################################################
##########################################################################################################
""" Adam """
# convert class vectors to binary class matrices

## LR model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

input_shape = list(x_train.shape[1:])

inputs = Input(shape=input_shape)
feature = Flatten()(inputs)
feature = Dense(2, activation='softmax')(feature)
model0_deb1 = Model(inputs=inputs, outputs=feature)

## Compile

monitor = 'val_auc'
patience = 10
mode = 'auto'
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'old_training_deb1.csv'))
early = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,mode=mode)
calls = [csv_logger, early]

loss = 'categorical_crossentropy' # 'mse'# 
optim = keras.optimizers.Adam()
#metrics_to_compute = [tf.keras.metrics.AUC(), 'acc']
metrics_to_compute = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'), # CategoricalAccuracy(name='accuracy'), # 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        ]
model0_deb1.compile(loss=loss,
              optimizer=Adam(parameters["learning_rate"]), # optim,
              metrics=metrics_to_compute)
          
## Training
history_deb1 = model0_deb1.fit(x_train, y_train,
          batch_size=parameters["batch_size"],
          epochs=parameters["epochs"],
          verbose=1,
          validation_data=(x_val,y_val), callbacks=calls)

training_acc = history_deb1.history['accuracy']
training_loss = history_deb1.history['loss']
val_acc = history_deb1.history['val_accuracy']
val_loss = history_deb1.history['val_loss']
training_AUC = history_deb1.history['auc']
val_AUC = history_deb1.history['val_auc']

score_MLR0_deb1 = model0_deb1.evaluate(x_test, y_test, verbose=1)
# score_MLR list corresponds to  model.metrics_names = ['loss', 'auc', 'acc']

y_train_pred = model0_deb1.predict(x_train)
y_val_pred = model0_deb1.predict(x_val)
y_test_pred = model0_deb1.predict(x_test)

y_train_pred =np.argmax(y_train_pred, axis = 1)
y_val_pred = np.argmax(y_val_pred, axis = 1)
y_test_pred = np.argmax(y_test_pred, axis = 1)

cm_train0_deb1 = confusion_matrix(y_train_1_column,y_train_pred, normalize = 'true')
cm_val0_deb1 = confusion_matrix(y_val_1_column,y_val_pred, normalize = 'true')
cm_test0_deb1 = confusion_matrix(y_test_1_column,y_test_pred, normalize = 'true')

with open(os.path.join(model_dir,'model0_deb1.txt'),'a') as f_conf:
    f_conf.write(str(model0_deb1.metrics_names) +'\n')
    f_conf.write('test score = '+ str(score_MLR0_deb1)+'\n')
    f_conf.write('cm_train1 = '+ str(cm_train0_deb1)+'\n')
    f_conf.write('cm_val1 = '+ str(cm_val0_deb1)+'\n')
    f_conf.write('cm_test1 = '+ str(cm_test0_deb1)+'\n')

##########################################################################################################
""" Adam + def categ acc """
# convert class vectors to binary class matrices

## LR model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

input_shape = list(x_train.shape[1:])

inputs = Input(shape=input_shape)
feature = Flatten()(inputs)
feature = Dense(2, activation='softmax')(feature)
model0_deb3 = Model(inputs=inputs, outputs=feature)

## Compile

monitor = 'val_auc'
patience = 10
mode = 'auto'
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'old_training_deb3.csv'))
early = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,mode=mode)
calls = [csv_logger, early]

loss = 'categorical_crossentropy' # 'mse'# 
optim = keras.optimizers.Adam()
#metrics_to_compute = [tf.keras.metrics.AUC(), 'acc']
metrics_to_compute = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'), # BinaryAccuracy(name='accuracy'), #
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        ]
model0_deb3.compile(loss=loss,
              optimizer=Adam(parameters["learning_rate"]), # optim,
              metrics=metrics_to_compute)
          
## Training
history_deb3 = model0_deb3.fit(x_train, y_train,
          batch_size=parameters["batch_size"],
          epochs=parameters["epochs"],
          verbose=1,
          validation_data=(x_val,y_val), callbacks=calls)

training_acc = history_deb3.history['accuracy']
training_loss = history_deb3.history['loss']
val_acc = history_deb3.history['val_accuracy']
val_loss = history_deb3.history['val_loss']
training_AUC = history_deb3.history['auc']
val_AUC = history_deb3.history['val_auc']

score_MLR0_deb3 = model0_deb3.evaluate(x_test, y_test, verbose=1)
# score_MLR list corresponds to  model.metrics_names = ['loss', 'auc', 'acc']

y_train_pred = model0_deb3.predict(x_train)
y_val_pred = model0_deb3.predict(x_val)
y_test_pred = model0_deb3.predict(x_test)

y_train_pred =np.argmax(y_train_pred, axis = 1)
y_val_pred = np.argmax(y_val_pred, axis = 1)
y_test_pred = np.argmax(y_test_pred, axis = 1)

cm_train0_deb3 = confusion_matrix(y_train_1_column,y_train_pred, normalize = 'true')
cm_val0_deb3 = confusion_matrix(y_val_1_column,y_val_pred, normalize = 'true')
cm_test0_deb3 = confusion_matrix(y_test_1_column,y_test_pred, normalize = 'true')

with open(os.path.join(model_dir,'model0_deb3.txt'),'a') as f_conf:
    f_conf.write(str(model0_deb3.metrics_names) +'\n')
    f_conf.write('test score = '+ str(score_MLR0_deb3)+'\n')
    f_conf.write('cm_train1 = '+ str(cm_train0_deb3)+'\n')
    f_conf.write('cm_val1 = '+ str(cm_val0_deb3)+'\n')
    f_conf.write('cm_test1 = '+ str(cm_test0_deb3)+'\n')

##########################################################################################################
""" Adam + def categ acc + def model """
# convert class vectors to binary class matrices

## LR model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

input_shape = list(x_train.shape[1:])

# inputs = Input(shape=input_shape)
# feature = Flatten()(inputs)
# feature = Dense(2, activation='softmax')(feature)
# model0_deb4 = Model(inputs=inputs, outputs=feature)

def mlr(num_classes, input_shape):
    ## Combined LR model
    egm_inputs = Input(shape=input_shape)
    egm_feature = Flatten()(egm_inputs)
    # egm_feature = Dense(num_classes, activation='softmax')(egm_feature)
    egm_feature = Dense(2, activation='softmax')(egm_feature)
    egm_model = Model(inputs=egm_inputs, outputs=egm_feature)

    egm_model.summary()
    return egm_model 

model0_deb4 = mlr(2, input_shape) 

## Compile
monitor = 'val_auc'
patience = 10
mode = 'auto'
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'old_training_deb4.csv'))
early = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,mode=mode)
calls = [csv_logger, early]

loss = 'categorical_crossentropy' # 'mse'# 
optim = keras.optimizers.Adam()
#metrics_to_compute = [tf.keras.metrics.AUC(), 'acc']
metrics_to_compute = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'), # BinaryAccuracy(name='accuracy'), #
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        ]
model0_deb4.compile(loss=loss,
              optimizer=Adam(parameters["learning_rate"]), # optim,
              metrics=metrics_to_compute)
          
## Training
history_deb4 = model0_deb4.fit(x_train, y_train,
          batch_size=parameters["batch_size"],
          epochs=parameters["epochs"],
          verbose=1,
          validation_data=(x_val,y_val), callbacks=calls)

training_acc = history_deb4.history['accuracy']
training_loss = history_deb4.history['loss']
val_acc = history_deb4.history['val_accuracy']
val_loss = history_deb4.history['val_loss']
training_AUC = history_deb4.history['auc']
val_AUC = history_deb4.history['val_auc']

score_MLR0_deb4 = model0_deb4.evaluate(x_test, y_test, verbose=1)
# score_MLR list corresponds to  model.metrics_names = ['loss', 'auc', 'acc']

y_train_pred = model0_deb4.predict(x_train)
y_val_pred = model0_deb4.predict(x_val)
y_test_pred = model0_deb4.predict(x_test)

y_train_pred =np.argmax(y_train_pred, axis = 1)
y_val_pred = np.argmax(y_val_pred, axis = 1)
y_test_pred = np.argmax(y_test_pred, axis = 1)

cm_train0_deb4 = confusion_matrix(y_train_1_column,y_train_pred, normalize = 'true')
cm_val0_deb4 = confusion_matrix(y_val_1_column,y_val_pred, normalize = 'true')
cm_test0_deb4 = confusion_matrix(y_test_1_column,y_test_pred, normalize = 'true')

with open(os.path.join(model_dir,'model0_deb4.txt'),'a') as f_conf:
    f_conf.write(str(model0_deb4.metrics_names) +'\n')
    f_conf.write('test score = '+ str(score_MLR0_deb4)+'\n')
    f_conf.write('cm_train1 = '+ str(cm_train0_deb4)+'\n')
    f_conf.write('cm_val1 = '+ str(cm_val0_deb4)+'\n')
    f_conf.write('cm_test1 = '+ str(cm_test0_deb4)+'\n')

print("debugged")

