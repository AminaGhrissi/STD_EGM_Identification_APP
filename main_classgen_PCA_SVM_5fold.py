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

## built in libraries
from tensorflow import keras
import scipy
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import imblearn as imbl
from math import ceil
import numpy as np 
from numpy import std, mean 
from operator import itemgetter 
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle as pl 
from sklearn.externals import joblib
import time 

## user's libraries 
from data_io import labels_of_files, adjust_batches_egms, adjust_batches, DataGenerator, data, load_data_generator, data_generator, get_f_names, fnames_split_train_val_test, balance_oversampling_files, read_data_for_ros
from train_graph import STD_classifier
from evaluate_model import plots, eval_and_store_gen, eval_and_store, eval_and_store_svm

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
        "architecture": "pca_svm", # "lenet", # "mlr", # "LSTM_VAVp", #"cnn_1D", # "lenet_drop", # 
        "VAVp": True, # False, # ,
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
        "dim": (1,2500),
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

## SVM parameters
kernel = 'rbf' # 'linear' # https://scikit-learn.org/stable/modules/svm.html
C = 1.0

## directories
model_dir = os.path.join('DL_STD_classif','classif',parameters["architecture"]+'_'+kernel+'_C'+str(int(C))+'_5fold') # '_generator_categ_5fold')
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
r = [None] * k

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

    ## PCA dimensionlity reduction 

    # define transform
    pca = PCA(n_components = 0.95, svd_solver = 'full')

    t = time.time()

    # prepare transform on dataset
    pca.fit(x_train)
    # save PCA model 
    pl.dump(pca, open(model_dir_kfold+"/pca_VAVp.pkl","wb"))

    # later reload the pickle file
    # pca_reload = pl.load(open(model_dir+"/pca_VAVp.pkl",'rb'))

    # apply transform to dataset
    x_t_train = pca.transform(x_train)  # x_t_train.shape = (43264, 285)
    x_t_test = pca.transform(x_test)

    r[fold_nb-1] = x_t_train.shape[1] # nb of PC kept: they explain enough variance

    ## SVM classification
    clf = SVC(kernel = kernel, C = C,  verbose = True)
    clf.fit(x_t_train, y_train)

    elapsed = time.time() - t 

    # save SVM model 
    joblib.dump(clf, model_dir_kfold+'/svm_linear.pkl') 

    # Load the pickle file
    # clf_load = joblib.load('.pkl') 

    test_acc[fold_nb-1], test_AUC[fold_nb-1], test_TPR[fold_nb-1], test_TNR[fold_nb-1], test_PPV[fold_nb-1], test_NPV[fold_nb-1], test_F1[fold_nb-1] = eval_and_store_svm(clf, model_dir_kfold, parameters["augment"], elapsed, x_t_train, y_train, x_t_test, y_test, r[fold_nb-1])

mean_test_acc = mean(test_acc)
mean_test_AUC = mean(test_AUC)
mean_test_TPR = mean(test_TPR)
mean_test_TNR = mean(test_TNR)
mean_test_PPV = mean(test_PPV)
mean_test_NPV = mean(test_NPV)
mean_test_F1 = mean(test_F1)
mean_r = mean(r)

stddev_test_acc = std(test_acc)
stddev_test_AUC = std(test_AUC)
stddev_test_TPR = std(test_TPR)
stddev_test_TNR = std(test_TNR)
stddev_test_PPV = std(test_PPV)
stddev_test_NPV = std(test_NPV)
stddev_test_F1 = std(test_F1)
stddev_r = std(r)

with open(os.path.join(model_dir,'performance_5fold.txt'),'a') as f_conf:
    f_conf.write('mean_r = ' + str(mean_r)+'\n')
    f_conf.write('stddev_r = ' + str(stddev_r)+'\n')

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

pca100 = PCA(n_components = 0.99, svd_solver = 'full')
pca100.fit(x_train)
exp_var = pca100.explained_variance_
perc_exp_var = pca100.explained_variance_ratio_
perc_cum_exp_var = np.cumsum(perc_exp_var) *100 # in percentage

cp_count = range(1,len(perc_exp_var)+1)

fig_handle = plt.figure(1)
plt.axhline(95,0,len(perc_exp_var), color='r', label='variance threshold')
plt.plot(cp_count, perc_cum_exp_var, 'b-')
plt.xlabel('PC')
plt.ylabel('Explained Variance (%)')
axes = plt.gca()
axes.set_xlim([1,len(perc_cum_exp_var)])
axes.set_ylim([0,100])
# plt.yticks(np.arange(0,100,5)) # encombrant
# plt.yticks(range(1), ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '100'])

# Set the tick labels font
for label in ([axes.title, axes.xaxis.label, axes.yaxis.label] + axes.get_xticklabels() + axes.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(18)
    label.set_fontweight('bold')

for line in axes.get_lines():
    line.set_linewidth(3)

plt.savefig(os.path.join(model_dir , 'explained_variance.png'), bbox_inches='tight')
plt.clf()

print('plotted')