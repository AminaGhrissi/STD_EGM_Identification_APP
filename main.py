# built in libraries
from tensorflow import keras
import scipy
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import imblearn as imbl

# user's libraries 
# from evalute_model import *
from data_io import data, load_data_generator
from train_graph import STD_classifier
from evaluate_model import plots, eval_and_store, eval_and_store_gen

# Seems to be useful for incremental memory allocation, otherwise TF consumes all memory of the GPU instantaniously
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

## DA - oversampling 
def balance_oversampling(x,y):
    X = np.reshape(x,[x.shape[0],-1],order='F')
    ros = imbl.over_sampling.RandomOverSampler(random_state=42)
    y = np.squeeze(y)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    reshape_dims = [X_resampled.shape[0]]+list(x.shape)[1:]
    return np.reshape(X_resampled, reshape_dims, order='F'),y_resampled

# metaparameters
parameters = {
        "is_it_CNN": False, # True, #
        "architecture": "mlr", # "lenet", #
        "n_folds": 5,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 100,
        "augment" : True, # False # ## oversampling"
        "monitor" : "val_auc",
        "patience" : 10,
        "mode" : "auto",
        "num_classes" : 2, 
        "steps_per_epoch" : 10,
        "validation_steps" : 10,
    }

# directories
model_dir = os.path.join('DL_STD_classif','classif',parameters["architecture"]+'_generator')
data_dir = os.path.join('DL_STD_classif',)

if parameters["augment"]:
    model_dir += '_ros'    
if not os.path.exists(model_dir): # Create target Directory if doesn't exist
    os.mkdir(model_dir)

# load data 
x,y = data(data_dir) 
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.50, random_state=42)

parameters["steps_per_epoch"] = x_train.shape[0] // parameters["batch_size"]
parameters["validation_steps"] = x_val.shape[0] // parameters["batch_size"]

if parameters["augment"]:
    print('ros training dataset')
    x_train, y_train = balance_oversampling(x_train,y_train)

if parameters["is_it_CNN"]:
    # Reshape x_train, x_val, x_test 
    x_train = np.expand_dims(x_train, axis=3)
    x_val = np.expand_dims(x_val, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

# convert class vectors to binary class matrices
y_train_1_column = y_train            # keep a copy of  y_test  before 1-hot transofmation -> useful for CM calculation
y_val_1_column = y_val            # keep a copy of  y_test  before 1-hot transofmation -> useful for CM calculation
y_test_1_column = y_test            # keep a copy of  y_test  before 1-hot transofmation -> useful for CM calculation

y_train = tf.keras.utils.to_categorical(y_train, 2)
y_val = tf.keras.utils.to_categorical(y_val, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

train_generator = load_data_generator(x_train,y_train, batch_size=parameters["batch_size"])
val_generator = load_data_generator(x_val,y_val, batch_size=parameters["batch_size"])
test_generator = load_data_generator(x_test,y_test, batch_size=parameters["batch_size"])

input_shape = list(x_train.shape[1:])

std_classifier = STD_classifier(parameters,input_shape, model_dir)
architecture = std_classifier.get_architecture()
model, history, elapsed = std_classifier.trainer_gen(architecture,std_classifier.call_functions(), train_generator, val_generator) # (x_val, y_val)) #, wandb)
# model, history, elapsed = std_classifier.trainer(architecture,std_classifier.call_functions(), x_train, y_train, x_val, y_val) #, wandb)
model.save(os.path.join(model_dir,'model.h5')) ## save model:
print("trained")

" works only if its input data is an interator: train_generator, val_generator, test_generator"
# eval_and_store_gen(model, model_dir, parameters["augment"], history, train_generator, y_train_1_column, val_generator, y_val_1_column, test_generator, y_test_1_column)
eval_and_store(model, model_dir, parameters["augment"], history, x_train, y_train, y_train_1_column, x_val, y_val, y_val_1_column, x_test, y_test, y_test_1_column)

plots(model_dir, history)

print("evaluated")

