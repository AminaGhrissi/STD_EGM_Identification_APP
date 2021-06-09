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
from data_io import data, load_data_generator, data_generator, get_f_names, fnames_split_train_val_test, balance_oversampling_files, read_data_for_ros
from train_graph import STD_classifier
from evaluate_model import plots, eval_and_store, eval_and_store_gen

# Seems to be useful for incremental memory allocation, otherwise TF consumes all memory of the GPU instantaniously
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# metaparameters
parameters = {
        "is_it_CNN2D": False, #True, #  
        "architecture": "mlr", # "lenet", # 
        "n_folds": 5,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 5,
        "augment" : True, #False, #  ## oversampling"
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

f_names = get_f_names()
train_files, val_files, test_files = fnames_split_train_val_test(f_names)

if parameters["augment"]:
    print('ros training dataset')
    train_files, y_train = balance_oversampling_files(train_files)

parameters["steps_per_epoch"] = len(train_files) // parameters["batch_size"]
parameters["validation_steps"] = len(val_files) // parameters["batch_size"] 

train_generator = data_generator(parameters, train_files , batch_size=parameters["batch_size"])
val_generator = data_generator(parameters, val_files, batch_size=parameters["batch_size"])
test_generator = data_generator(parameters, test_files, batch_size=parameters["batch_size"])

if parameters["is_it_CNN2D"]:
    input_shape = (2500, 12, 1)
else:
    input_shape = (2500, 12) # list(x_train.shape[1:])

std_classifier = STD_classifier(parameters,input_shape, model_dir)
architecture = std_classifier.get_architecture()
model, history, elapsed = std_classifier.trainer_rand_gen(architecture,std_classifier.call_functions(), train_generator, val_generator) # (x_val, y_val)) #, wandb)
# model, history, elapsed = std_classifier.trainer(architecture,std_classifier.call_functions(), x_train, y_train, x_val, y_val) #, wandb)
model.save(os.path.join(model_dir,'model.h5')) ## save model:
print("trained")

" works only if its input data is an interator: train_generator, val_generator, test_generator"
# eval_and_store_gen(model, model_dir, parameters["augment"], history, train_generator, y_train_1_column, val_generator, y_val_1_column, test_generator, y_test_1_column)
# eval_and_store(model, model_dir, parameters["augment"], history, x_train, y_train, y_train_1_column, x_val, y_val, y_val_1_column, x_test, y_test, y_test_1_column)
eval_and_store_gen(model, model_dir, parameters["augment"], history, test_generator, elapsed)
plots(model_dir, history)

print("evaluated")

