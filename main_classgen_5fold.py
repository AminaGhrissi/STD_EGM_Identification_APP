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
from data_io import labels_of_files, adjust_batches_egms, DataGenerator, data, load_data_generator, data_generator, get_f_names, fnames_split_train_val_test, balance_oversampling_files, read_data_for_ros
from train_graph import STD_classifier
from evaluate_model import plots, eval_and_store_gen, eval_and_store
from math import ceil
import numpy as np 
from numpy import std, mean 
from operator import itemgetter
from sklearn.model_selection import StratifiedKFold

## Seems to be useful for incremental memory allocation, otherwise TF consumes all memory of the GPU instantaniously
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

## metaparameters
parameters = {
        "is_it_CNN2D": False, #True, # 
        "is_it_CNN1D": False, # True, # 
        "is_it_CNN3D": True,  # False, # 
        "architecture": "VGG16_EGM", # "mlr",  #"lenet", # "LSTM_VAVp", #"lenet_drop", # "cnn_1D",# 
        "VAVp": False, #True, #  
        "image": False, # True, # 
        "image3D": True, # False
        "tensorize": False, # True, #
        "n_folds": 5, # 2,  
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 1, # 50, #
        "augment" : True, # False,  #  ## oversampling"
        "monitor" : "val_loss", #"val_auc",
        "patience" : 10,
        "mode" : "auto",
        "num_classes" : 2, 
        "steps_per_epoch" : None, #10,
        "validation_steps" : None, #10,
        "dim": (389, 515, 3), # (389, 515), # (2500,12), # #(2500, 1), # (1, 2500, 12), #   (10, 10, 4), # 
        "shuffle": False, # True,
    }

# add 4th dim if 2D CNN
input_shape = parameters["dim"]
if (parameters["is_it_CNN2D"] or parameters["is_it_CNN1D"]) and not (parameters["tensorize"] or parameters["VAVp"] or parameters["is_it_CNN3D"]):
    input_shape = list(input_shape)
    input_shape.append(1)
    input_shape = tuple(input_shape)

## directories
model_dir = os.path.join('DL_STD_classif','classif','optim_'+parameters["architecture"]+'_5fold') # '_generator_categ_5fold')
data_dir = os.path.join('DL_STD_classif',)

if parameters["augment"]:
    model_dir += '_ros' 

if parameters["image"]:
    model_dir += '_BINimg' 

if parameters["image3D"]:
    model_dir += '_3D_BINimg'

if parameters["tensorize"]:
    model_dir += '_tensorize' 

if parameters["VAVp"]:
    model_dir += '_VAVp' 

if not os.path.exists(model_dir): # Create target Directory if doesn't exist
    os.mkdir(model_dir)

## split files_names 
f_names = get_f_names()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
k = parameters["n_folds"]
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

y_f_names = labels_of_files(f_names)

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
        train_files, y_train = balance_oversampling_files(train_files, parameters["VAVp"], parameters["image"], parameters["image3D"], parameters["tensorize"])
        nb_train_aug = len(train_files)
        p = np.random.permutation(nb_train_aug)
        train_files = [train_files[i] for i in p]
        y_train = y_train[p]
        
    print('adjust train')
    train_files, y_train = adjust_batches_egms(train_files, parameters["batch_size"], parameters["VAVp"], parameters["image"], parameters["image3D"], parameters["tensorize"])
    print('adjust val')
    val_files, y_val = adjust_batches_egms(val_files, parameters["batch_size"], parameters["VAVp"], parameters["image"], parameters["image3D"], parameters["tensorize"])
    print('adjust test')
    test_files, y_test = adjust_batches_egms(test_files, parameters["batch_size"], parameters["VAVp"], parameters["image"], parameters["image3D"], parameters["tensorize"])

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

    # Built & run graph
    std_classifier = STD_classifier(parameters,input_shape, model_dir_kfold)
    architecture = std_classifier.get_architecture()
    print('start training')
    model, history, elapsed = std_classifier.trainer_gen(architecture,std_classifier.call_functions(), train_generator, val_generator) # (x_val, y_val)) #, wandb)
    # model, history, elapsed = std_classifier.trainer(architecture,std_classifier.call_functions(), x_train, y_train, x_val, y_val) #, wandb)
    print("trained")

    model.save(os.path.join(model_dir_kfold,'model.h5'))

    " works only if its input data is an interator: train_generator, val_generator, test_generator"
    history = history.history
    #eval_and_store_gen(model, model_dir, parameters["augment"], history, test_generator, elapsed)
    # eval_and_store(model, model_dir_kfold, parameters["augment"], history, elapsed, train_generator, y_train, val_generator, y_val, test_generator, y_test)

    test_acc[fold_nb-1], test_AUC[fold_nb-1], test_TPR[fold_nb-1], test_TNR[fold_nb-1], test_PPV[fold_nb-1], test_NPV[fold_nb-1], test_F1[fold_nb-1] = eval_and_store(model, model_dir_kfold, parameters["augment"], history, elapsed, train_generator, y_train, val_generator, y_val, test_generator, y_test)
    plots(model_dir_kfold, history)
    del model, history, train_generator, val_generator, test_generator, train_files, val_files, test_files, y_train, y_val, y_test

mean_test_acc = mean(test_acc)
mean_test_AUC = mean(test_AUC)
mean_test_TPR = mean(test_TPR)
mean_test_TNR = mean(test_TNR)
mean_test_PPV = mean(test_PPV)
mean_test_NPV = mean(test_NPV)
mean_test_F1 = mean(test_F1)

stddev_test_acc = std(test_acc)
stddev_test_AUC = std(test_AUC)
stddev_test_TPR = std(test_TPR)
stddev_test_TNR = std(test_TNR)
stddev_test_PPV = std(test_PPV)
stddev_test_NPV = std(test_NPV)
stddev_test_F1 = std(test_F1)

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


