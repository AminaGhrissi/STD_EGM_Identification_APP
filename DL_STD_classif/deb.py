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
import pickle as pl 

from sklearn.model_selection import StratifiedKFold

PIK_ifes = os.path.join('DL_STD_classif', 'f_names.dat')

with open(PIK_ifes, "rb") as f:
    f_names_ifes = pl.load(f)
    y_f_names_ifes = pl.load(f)


## metaparameters
parameters = {
        "is_it_CNN2D": False, # True, # 
        "is_it_CNN1D": False, # True, # 
        "architecture": "mlr", # "lenet", # "mlr", # "LSTM_VAVp", #"cnn_1D", # "lenet_drop", # 
        "VAVp": False, # True,
        "n_folds": 5,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 2, #50,
        "augment" : False, # True, #    ## oversampling"
        "monitor" : "val_auc", # "val_loss", #
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

## split files_names 
f_names = get_f_names()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
k = 5
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

y_f_names, pt_name = labels_of_files(f_names)

PIK = os.path.join(model_dir, 'f_names.dat')
with open(PIK, 'wb') as f:
    pl.dump(f_names, f)
    pl.dump(y_f_names, f)  

print('loaded')


# sorting both the lists 
f_names.sort() 
f_names_ifes.sort() 
  
# using == to check if  
# lists are equal 
if f_names == f_names_ifes: 
    print ("The lists are identical") 
else : 
    print ("The lists are not identical") 


if y_f_names == y_f_names_ifes: 
    print ("The y lists are identical") 
else : 
    print ("The y lists are not identical") 

"""
import pickle as pl
with open(PIK, "rb") as f:
    f_names = pl.load(f)
    y_f_name = pl.load(f)
"""