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

## Seems to be useful for incremental memory allocation, otherwise TF consumes all memory of the GPU instantaniously
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

## metaparameters
parameters = {
        "is_it_CNN2D": True, # False, #
        "is_it_CNN1D": False, # True, # 
        "architecture": "lenet", # "mlr", # "LSTM_VAVp", #"cnn_1D", # "lenet_drop", # 
        "VAVp": False, # True,
        "n_folds": 5,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 2, # 50,
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
model_dir = os.path.join('DL_STD_classif','classif',parameters["architecture"]+'_5fold_test') # '_generator_categ_5fold')
data_dir = os.path.join('DL_STD_classif',)

if parameters["augment"]:
    model_dir += '_ros'    
if not os.path.exists(model_dir): # Create target Directory if doesn't exist
    os.mkdir(model_dir)

## split files_names 
f_names = get_f_names()

from sklearn.model_selection import StratifiedKFold

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
k = 5
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

y_f_names, pt_name = labels_of_files(f_names)

fold_nb = 0

test_acc = [None] * k
test_AUC = [None] * k
test_TPR = [None] * k
test_TNR = [None] * k
test_PPV = [None] * k
test_NPV = [None] * k
test_F1 = [None] * k

for train_val, test in kfold.split(f_names, y_f_names):
    train_val_files = list(itemgetter(*train_val)(f_names))
    test_files = list(itemgetter(*test)(f_names))

    train_files, val_files = train_test_split(train_val_files, test_size=0.20, random_state=seed)

    fold_nb += 1
    model_dir_kfold = os.path.join(model_dir, 'fold'+str(fold_nb))
    if not os.path.exists(model_dir_kfold): # Create target Directory if doesn't exist
        os.mkdir(model_dir_kfold)

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


    break

def check_nan(D,Y):
    pt_nans = []
    y_nans = []
    for i in range(D.shape[0]):
        pt = D[i,:,:,0]
        array_sum = np.sum(pt)
        array_has_nan = np.all(np.isfinite(array_sum)) # array_sum.dtype == 'float32' # np.isnan(array_sum)
        if not array_has_nan:
            pt_nans = pt_nans + [i]

    y_has_nan = np.sum(Y).dtype == 'float64' # np.isnan(np.sum(Y))
    if not y_has_nan:
        y_nans = y_nans + [i]
    return pt_nans # y_nans # 

x_train_nans = check_nan(x_train, y_train)
x_val_nans = check_nan(x_val, y_val)
x_test_nans = check_nan(x_test, y_test)

print('x_train_nans')
print(x_train_nans)

print('x_val_nans')
print(x_val_nans)

print('x_test_nans')
print(x_test_nans)

print([train_files[10567]]) # ['pat_36_pt4128.pickle']

"""
Isfinite
x_train_nans
[10567]
x_val_nans
[]
x_test_nans
[]
"""