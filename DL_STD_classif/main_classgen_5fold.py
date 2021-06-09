#################################################################################
#
#               University of Nice Côte d’Azur (UCA) -
#               National Center of Scientific Research (CNRS) –
#               CHU of Nice (CHUN) -
#               Instituto Federal do Espirito Santo (IFES)
#              Copyright © 2020 UCA, CNRS, CHUN, IFES  All Rights Reserved.
#
#     These computer program listings and specifications, herein, are
#     the property of UCA, CNRS, CHUN and IFES  
#     shall not be reproduced or copied or used in whole or in part as
#     the basis for manufacture or sale of items without written permission.
#     For a license agreement, please contact: https:#www.sattse.com/  
#
#################################################################################

## Built-in libraries
import tensorflow as tf
from tensorflow import keras
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import imblearn as imbl
from math import ceil
import numpy as np 
from numpy import std, mean 
from operator import itemgetter
from sklearn.model_selection import StratifiedKFold

## Implemented libraries 
from data_io import labels_of_files, adjust_batches_egms, DataGenerator, data, data_generator, get_f_names, balance_oversampling_files
from train_graph import STD_classifier
from evaluate_model import plots, eval_and_store

## Seems to be useful for incremental memory allocation, otherwise TF consumes all memory of the GPU instantaniously
"""
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
"""

## Set random seed for reproducibility
seed = 7
np.random.seed(seed)

## Set metaparameters dictionary
parameters = {
        "is_it_CNN1D": False, ## boolean variable stating whether the model is a 1D CNN or not    
        "is_it_CNN2D": False, ## boolean variable stating whether the model is a 2D CNN or not
        "is_it_CNN3D": True,  ## boolean variable stating whether the model is a 3D CNN or not
        "architecture": "VGG16_EGM", ## name of the model architecture, can take: "pca_svm", "mlr",  "lenet", "LSTM_VAVp", "lenet_drop", "cnn_1D" or  "VGG16_EGM" 
        "VAVp": False, ## boolean variable stating whether the input format is VAVp or not  
        "image": False, ## boolean variable stating whether the input format is a (2D) image or not  
        "image3D": True, ## boolean variable stating whether the input format is a 3D image or not  
        "tensorize": False, ## boolean variable stating whether to tensorize input the 10-channel EGM sample or not  
        "n_folds": 5,       ## number of folds in k-fold cross-validation  
        "learning_rate": 0.001, ## ad-hoc choice of the learning rate used in gradient descent optimization algorithm
        "batch_size": 32,       ## mini batch size for mini-batch gradient descent optimization algorithm, can be increased if more memory is allowed
        "epochs": 1,            ## default used 100, max number of epochs in mini-batch gradient descent optimization algorithm
        "augment" : True,       ## boolean variable stating whether to augment with oversampling or not 
        "monitor" : "val_loss", ## early stopping criterion of the training algorithm, can take: "val_auc", "val_loss" ...
        "patience" : 10,        ## number of epochs to wait before early stop if no progress on the validation set
        "mode" : "auto",        ## stopping mode, can take: max, min, auto
        "num_classes" : 2,      ## number of classes: STD vs. non-STD
        "steps_per_epoch" : None, ## number of mini batchesto yield from generators before finishing the current epoch and moving to the next, None takes all data
        "validation_steps" : None, ## steps_per_epoch for the validation generator
        "dim": (389, 515, 3), ## dimensions of the transformed data to be processed by the model, depends on the model and the input format, can take: (389, 515), # (2500,12), # (2500, 1), # (1, 2500, 12), # (10, 10, 4), # (389, 515, 3) 
        "shuffle": False, ## No shuffling of data within each mini batch for tracability 
    }

# Add 4th dim to input data for 1D, 2D CNN models
input_shape = parameters["dim"]
if (parameters["is_it_CNN2D"] or parameters["is_it_CNN1D"]) and not (parameters["tensorize"] or parameters["VAVp"] or parameters["is_it_CNN3D"]):
    input_shape = list(input_shape)
    input_shape.append(1)
    input_shape = tuple(input_shape)

import pdb; pdb.set_trace() ## for debugging

## Directories
model_dir = os.path.join('DL_STD_classif','classif','optim_'+parameters["architecture"]+'_5fold')
data_dir = os.path.join('DL_STD_classif',)

## Update filenames with respect to the settings
if parameters["augment"]:
    model_dir += '_ros' 

if parameters["image"]:
    model_dir += '_BINimg' ## BIN for binary 

if parameters["image3D"]:
    model_dir += '_3D_BINimg'

if parameters["tensorize"]:
    model_dir += '_tensorize' 

if parameters["VAVp"]: ## VAVp stands for a 1D time series called: Maximal Voltage Absolute Values at any of the PentaRay bipoles (VAVp) 
    model_dir += '_VAVp' 

if not os.path.exists(model_dir): # Create target Directory if doesn't exist
    os.mkdir(model_dir)

## Get filenames and stote them in f_names variable
f_names = get_f_names()

## Get labels corresponding to f_names data points and store them in y_f_names variable
y_f_names = labels_of_files(f_names)

## Kfold setting
k = parameters["n_folds"]
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
fold_nb = 0

## Initialize empty lists where to store performance metrics over the k folds
test_acc = [None] * k
test_AUC = [None] * k
test_TPR = [None] * k
test_TNR = [None] * k
test_PPV = [None] * k
test_NPV = [None] * k
test_F1 = [None] * k

## Loop over Kfolds
for train_val, test in kfold.split(f_names, y_f_names):
    # Split data (samples+labels) into training (train), validation (val) and test sets
    train_val_files = list(itemgetter(*train_val)(f_names))
    test_files = list(itemgetter(*test)(f_names))
    train_files, val_files = train_test_split(train_val_files, test_size=0.20, random_state=seed)

    fold_nb += 1
    # Create target Directory for each fold, if doesn't exist
    model_dir_kfold = os.path.join(model_dir, 'fold'+str(fold_nb))
    if not os.path.exists(model_dir_kfold): 
        os.mkdir(model_dir_kfold)

    ## Oversampling: augment only the train dataset with Oversampling method
    if parameters["augment"]:
        print('ros training dataset')
        train_files, y_train = balance_oversampling_files(train_files, parameters["VAVp"], parameters["image"], parameters["image3D"], parameters["tensorize"])
        nb_train_aug = len(train_files)
        p = np.random.permutation(nb_train_aug)
        train_files = [train_files[i] for i in p]
        y_train = y_train[p]

    ##  Adjust mini batches so that all batches contain the same number of samples
    print('adjust train')
    train_files, y_train = adjust_batches_egms(train_files, parameters["batch_size"], parameters["VAVp"], parameters["image"], parameters["image3D"], parameters["tensorize"])
    print('adjust val')
    val_files, y_val = adjust_batches_egms(val_files, parameters["batch_size"], parameters["VAVp"], parameters["image"], parameters["image3D"], parameters["tensorize"])
    print('adjust test')
    test_files, y_test = adjust_batches_egms(test_files, parameters["batch_size"], parameters["VAVp"], parameters["image"], parameters["image3D"], parameters["tensorize"])

    # Dictionary with partitioned data 
    partition = {
        'train': train_files, 
        'validation': val_files,
        'test' : test_files,  
        }
    
    ## Generators
    print('generator train')
    train_generator = DataGenerator(partition['train'], parameters)
    print('generator val')
    val_generator = DataGenerator(partition['validation'], parameters)
    print('generator test')
    test_generator = DataGenerator(partition['test'], parameters)

    # Built & run graph; train models with generators
    std_classifier = STD_classifier(parameters,input_shape, model_dir_kfold)
    architecture = std_classifier.get_architecture()
    print('start training')
    model, history, elapsed = std_classifier.trainer_gen(architecture,std_classifier.call_functions(), train_generator, val_generator)
    print("trained")

    model.save(os.path.join(model_dir_kfold,'model.h5'))

    ## Evalute model 
    " works only if its input data is an interator: train_generator, val_generator, test_generator"
    history = history.history
    test_acc[fold_nb-1], test_AUC[fold_nb-1], test_TPR[fold_nb-1], test_TNR[fold_nb-1], test_PPV[fold_nb-1], test_NPV[fold_nb-1], test_F1[fold_nb-1] = eval_and_store(model, model_dir_kfold, parameters["augment"], history, elapsed, train_generator, y_train, val_generator, y_val, test_generator, y_test)
    plots(model_dir_kfold, history)

    ## Free unnecessary memory
    del model, history, train_generator, val_generator, test_generator, train_files, val_files, test_files, y_train, y_val, y_test

## Average performance over Kfold CV
# mean = average
mean_test_acc = mean(test_acc)
mean_test_AUC = mean(test_AUC)
mean_test_TPR = mean(test_TPR)
mean_test_TNR = mean(test_TNR)
mean_test_PPV = mean(test_PPV)
mean_test_NPV = mean(test_NPV)
mean_test_F1 = mean(test_F1)
# stddev = standard deviation
stddev_test_acc = std(test_acc)
stddev_test_AUC = std(test_AUC)
stddev_test_TPR = std(test_TPR)
stddev_test_TNR = std(test_TNR)
stddev_test_PPV = std(test_PPV)
stddev_test_NPV = std(test_NPV)
stddev_test_F1 = std(test_F1)

## Save the average (+- standard deviation) performance metrics over 
with open(os.path.join(model_dir,'performance_5fold.txt'),'a') as f_conf:
    f_conf.write('mean_test_acc = ' + str(mean_test_acc)+'\n')
    f_conf.write('mean_test_AUC = ' + str(mean_test_AUC)+'\n')
    f_conf.write('mean_test_TPR = ' + str(mean_test_TPR)+'\n')
    f_conf.write('mean_test_TNR = ' + str(mean_test_TNR)+'\n')
    f_conf.write('mean_test_PPV = ' + str(mean_test_PPV)+'\n')
    f_conf.write('mean_test_NPV = ' + str(mean_test_NPV)+'\n')
    f_conf.write('mean_test_F1 = ' + str(mean_test_F1)+'\n')

    f_conf.write('stddev_test_acc = ' + str(stddev_test_acc)+'\n')
    f_conf.write('stddev_test_AUC = ' + str(stddev_test_AUC)+'\n')
    f_conf.write('stddev_test_TPR = ' + str(stddev_test_TPR)+'\n')
    f_conf.write('stddev_test_TNR = ' + str(stddev_test_TNR)+'\n')
    f_conf.write('stddev_test_PPV = ' + str(stddev_test_PPV)+'\n')
    f_conf.write('stddev_test_NPV = ' + str(stddev_test_NPV)+'\n')
    f_conf.write('stddev_test_F1 = ' + str(stddev_test_F1)+'\n')

print("evaluated")


