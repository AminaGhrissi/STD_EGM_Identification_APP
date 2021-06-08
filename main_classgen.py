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
from evaluate_model import plots, eval_and_store_gen, eval_and_store
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
        "is_it_CNN1D": False, # True, # 
        "architecture": "mlr", # "LSTM_VAVp", #"cnn_1D", # "lenet_drop", # "lenet", # 
        "VAVp": False, # True,
        "n_folds": 5,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "augment" : True, # False,  #   ## oversampling"
        "monitor" : "val_loss", #"val_auc",
        "patience" : 10,
        "mode" : "auto",
        "num_classes" : 2, 
        "steps_per_epoch" : None, #10,
        "validation_steps" : None, #10,
        "dim": (2500,1),
        "shuffle": False, # True,
    }
    
# add 4th dim if 2D CNN
if parameters["is_it_CNN2D"]:
    input_shape = (2500, 12, 1)
else:
    input_shape = (2500, 12) # list(x_train.shape[1:])
if parameters["VAVp"]:
    parameters["dim"] = (2500,1)
    input_shape = (2500,1)

## directories
model_dir = os.path.join('DL_STD_classif','classif',parameters["architecture"]+'_generator_categ_VAVp')
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
    train_files, y_train = balance_oversampling_files(train_files, parameters["VAVp"])
    nb_train_aug = len(train_files)
    p = np.random.permutation(nb_train_aug)
    train_files = [train_files[i] for i in p]
    y_train = y_train[p]

train_files, y_train, x_train = adjust_batches_egms(train_files, parameters["batch_size"], parameters["VAVp"])
val_files, y_val, x_val = adjust_batches_egms(val_files, parameters["batch_size"], parameters["VAVp"])
test_files, y_test, x_test = adjust_batches_egms(test_files, parameters["batch_size"], parameters["VAVp"])

if parameters["is_it_CNN2D"] or parameters["is_it_CNN1D"]:
    # Reshape x_train, x_val, x_test in 4D tensor if CNN 2D
    x_train = np.expand_dims(x_train, axis=3)
    x_val = np.expand_dims(x_val, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

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
#eval_and_store_gen(model, model_dir, parameters["augment"], history, test_generator, elapsed)
eval_and_store(model, model_dir, parameters["augment"], history, elapsed, x_train, y_train, x_val, y_val, x_test, y_test, test_generator)

plots(model_dir, history)
print("evaluated")


