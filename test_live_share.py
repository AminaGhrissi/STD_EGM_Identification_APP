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
from data_io import labels_of_files, adjust_batches_egms, adjust_batches, DataGenerator, data, load_data_generator, data_generator, get_f_names, fnames_split_train_val_test, balance_oversampling_files, read_data_for_ros
from train_graph import STD_classifier
from evaluate_model import plots, eval_and_store_gen, eval_and_store
from math import ceil
import numpy as np 
from numpy import std, mean 
from operator import itemgetter 

# ## Seems to be useful for incremental memory allocation, otherwise TF consumes all memory of the GPU instantaniously
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

## metaparameters
parameters = {
        "is_it_CNN2D": True, # False, # 
        "is_it_CNN1D": False, # True, # 
        "architecture": "lenet", # "mlr", # "LSTM_VAVp", #"cnn_1D", # "lenet_drop", # 
        "VAVp": False, # True,
        "n_folds": 5,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 2, #50,
        "augment" : True, # False,  #   ## oversampling"
        "monitor" : "val_loss", #"val_auc",
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
if parameters["VAVp"]:
    parameters["dim"] = (2500,1)
    input_shape = (2500,1)

## directories
model_dir = os.path.join('DL_STD_classif','classif',parameters["architecture"]+'_5fold') # '_generator_categ_5fold')
data_dir = os.path.join('DL_STD_classif',)

if parameters["augment"]:
    model_dir += '_ros'    
if not os.path.exists(model_dir): # Create target Directory if doesn't exist
    os.mkdir(model_dir)