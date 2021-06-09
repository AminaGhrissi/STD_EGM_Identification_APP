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
##                    Functions to read data
#################################################################################

## Built-in libraries
from scipy.io import loadmat
import numpy as np 
import pickle as pl
import matplotlib.pyplot as plt
import time 
import os
from scipy.spatial.distance import cdist
from scipy.io import loadmat
import pandas as pd
import imblearn as imbl
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from math import ceil 
random.seed(42) ## set random seed for reproducibility

def data(data_dir):
    f_name_patients = 'Ablation_prediction_STD_Python/Ablation_labeling/Ablation_patients.pickle'
    with open(f_name_patients, 'rb') as ff:
        XYZ_penta_patients = pl.load(ff)
        egms_patients = pl.load(ff)
        labels_STD_patients = pl.load(ff)
        labels_abl_patients = pl.load(ff)
        nb_points_patients = pl.load(ff)
        nb_abl_points_patients = pl.load(ff)
        D_nearest_patients = pl.load(ff)
        nb_points_pentaray_patients = pl.load(ff)
        nb_sessions_patients = pl.load(ff)
    print('loaded')

    # Reshape
    signals = egms_patients
    nb_points = signals.shape[0]
    x = np.zeros((nb_points,signals[0].shape[1],signals[0].shape[0]))

    y = labels_STD_patients

    for pt in range(nb_points):
        x[pt,:,:] = np.transpose(signals[pt])
    x = x.astype('float16')

    return x, y 

def load_data_generator(x_train, y_train, batch_size):
    signals_order = list(range(x_train.shape[0]))
    random.shuffle(signals_order)

    for step in range(0, len(signals_order), batch_size):
        # batch_paths = signals_order[step:step + batch_size]
        if len(signals_order) > step + batch_size:
            batch_paths = signals_order[step:step + batch_size]
        else:
            batch_paths = signals_order[-batch_size:]
        
        x_train_batch = x_train[batch_paths,:]
        y_train_batch = y_train[batch_paths]

        x_train_batch = np.asarray(x_train_batch)
        y_train_batch = np.asarray(y_train_batch)

        yield (x_train_batch, y_train_batch)

## Returns a list of the filenames in 'data' folder, each data point contains a the 10-channel EGM record 
def get_f_names():
    # Outputs: 
    # f_names = list with data labels
    pt_path = os.path.join('data')
    f_names = os.listdir(pt_path)

    return f_names

def tensorize_egm_matrix(egm_matrix):
    # egm_tensor = np.reshape(egm_matrix, (1,2500, 12), order= 'C' and 'F') 
    egm_tensor = np.empty([1,egm_matrix.shape[1], egm_matrix.shape[0]])
    for l in range(egm_matrix.shape[0]):
        egm_tensor[0,:, l] = egm_matrix[l,:]
    return egm_tensor

def read_pt(pt_name, VAVp, image, image3D, tensorize):
    pt_name_nopath = pt_name
    if image:
        img_path = os.path.join('Bin_EGM_plots1')
        img_name = os.path.join(img_path, pt_name_nopath.split(".")[0]+'.png')
        egm_pat_pt_load = plt.imread(img_name)

        labels_path = os.path.join('labels')
        labels_name = os.path.join(labels_path,pt_name)
        with open(labels_name, 'rb') as ff:
            label_STD_pat_pat_load = pl.load(ff)
    elif image3D:
        img_path = os.path.join('EGM_plots_3slices')
        img_name = os.path.join(img_path, pt_name_nopath.split(".")[0]+'.dat')
        egm_pat_pt_load = np.load(img_name) # egm_pat_pt_load_slice

        labels_path = os.path.join('labels')
        labels_name = os.path.join(labels_path,pt_name)
        with open(labels_name, 'rb') as ff:
            label_STD_pat_pat_load = pl.load(ff)
    else:
        pt_path = os.path.join('data')
        pt_name = os.path.join(pt_path,pt_name)
        with open(pt_name, 'rb') as ff:
            egm_pat_pt_load = pl.load(ff)
            label_STD_pat_pat_load = pl.load(ff)
        if VAVp:
            egm_pat_pt_load = np.max(egm_pat_pt_load, axis=0)
    
        elif tensorize:
            egm_pat_pt_load = tensorize_egm_matrix(egm_pat_pt_load)

    if not (image or tensorize or image3D) :
        egm_pat_pt_load = np.transpose(egm_pat_pt_load)
    egm_pat_pt_load = egm_pat_pt_load.astype('float16')


    return egm_pat_pt_load, label_STD_pat_pat_load

## Get labels corresponding to f_names data points
def labels_of_files(f_names):
    # Inputs:
    # f_names: list with EGM sample names
    # Outputs: 
    # labels : list with the corresponding labels STD vs. non-STD
    labels = []
    for pt_name in f_names:
        labels_path = os.path.join('labels')
        pt_name = os.path.join(labels_path,pt_name)
        with open(pt_name, 'rb') as ff:
            label_STD_pat_pat_load = pl.load(ff)            
        labels.append(label_STD_pat_pat_load)
    return labels

def fnames_split_train_val_test(f_names):
    # Split train and test files
    train_files, test_val_files = train_test_split(f_names, test_size=0.30, random_state=342)
    val_files, test_files = train_test_split(test_val_files, test_size=0.50, random_state=342)
    
    return train_files, val_files, test_files

def data_generator(parameters, f_names, batch_size):
    VAVp = parameters["VAVp"]
    image = parameters["image"]
    image3D = parameters["image3D"]
    tensorize = parameters["tensorize"]
    " random mini batches - trial coz need infinte loop"
    while True:
        batch_paths = np.random.choice(a=f_names, size=batch_size)
        arrays = []
        labels = []
        for input_path in batch_paths:
            x, y = read_pt(input_path, VAVp, image, image3D, tensorize)
            arrays.append(x.astype('float16'))
            labels.append(y)

        arrays = np.asarray(arrays)
        labels = np.asarray(labels)
        labels = tf.keras.utils.to_categorical(labels, 2)

        if parameters["is_it_CNN2D"]:
            # Reshape x_train, x_val, x_test if CNN 2D
            arrays = np.expand_dims(arrays, axis=3)

        yield arrays, labels

## Oversampling 
def balance_oversampling_files(x, VAVp, image, image3D, tensorize):
    # Inputs:
    # x = filenames of data samples to be augmented
    # VAVp = boolean variable stating whether the input format is VAVp or not  
    # image = boolean variable stating whether the input format is a (2D) image or not  
    # image3D = boolean variable stating whether the input format is a 3D image or not  
    # tensorize = boolean variable stating whether to tensorize input the 10-channel EGM sample or not  
    # Outputs: 
    # X_resampled = augmented x
    # y_resampled = corresponding augmented lables
    y = read_data_for_ros(x, VAVp, image, image3D, tensorize)
    x = np.asarray(x)
    x = np.expand_dims(x, axis=1)
    ros = imbl.over_sampling.RandomOverSampler(random_state=42)
    y = np.squeeze(y)
    X_resampled, y_resampled = ros.fit_resample(x, y)
    X_resampled = np.squeeze(X_resampled)
    X_resampled = X_resampled.tolist()

    return X_resampled ,y_resampled


def read_data_for_ros(f_names, VAVp, image, image3D, tensorize):
    # Inputs:
    # f_names = filenames of data samples to be augmented
    # VAVp = boolean variable stating whether the input format is VAVp or not  
    # image = boolean variable stating whether the input format is a (2D) image or not  
    # image3D = boolean variable stating whether the input format is a 3D image or not  
    # tensorize = boolean variable stating whether to tensorize input the 10-channel EGM sample or not  
    # Outputs: 
    # labels = list of the corresponding labels
    labels = labels_of_files(f_names)
    labels = np.asarray(labels)
    return labels

##  Adjust mini batches so that all batches contain the same number of samples
def adjust_batches_egms(f_names, batch_size, VAVp, image, image3D, tensorize):
    # Inputs:
    # f_names = filenames of data samples
    # batch_size = mini batch_size
    # VAVp = boolean variable stating whether the input format is VAVp or not  
    # image = boolean variable stating whether the input format is a (2D) image or not  
    # image3D = boolean variable stating whether the input format is a 3D image or not  
    # tensorize = boolean variable stating whether to tensorize input the 10-channel EGM sample or not  
    # Outputs:    
    # f_names = f_names ensuring all mini batches are equal
    # labels = corresponding labels

    nb_samples = len(f_names)
    labels = labels_of_files(f_names)
        
    if nb_samples % batch_size:
        nb_comlementary_samples = batch_size - nb_samples % batch_size
        comlementary_samples = np.random.randint(nb_samples, size=(nb_comlementary_samples))
        complementry_f_names = [f_names[i] for i in comlementary_samples]
        complementry_labels = [labels[i] for i in comlementary_samples]

        f_names = f_names + complementry_f_names
        labels = labels + complementry_labels
        
    labels = np.asarray(labels)

    return f_names, labels 

##  Adjust mini batches so that all batches contain the same number of samples
def adjust_batches_egms_SVM(f_names, batch_size, VAVp, image, image3D, tensorize):
    # Inputs:
    # f_names = filenames of data samples
    # batch_size = mini batch_size
    # VAVp = boolean variable stating whether the input format is VAVp or not  
    # image = boolean variable stating whether the input format is a (2D) image or not  
    # image3D = boolean variable stating whether the input format is a 3D image or not  
    # tensorize = boolean variable stating whether to tensorize input the 10-channel EGM sample or not  
    # Outputs:    
    # f_names = f_names ensuring all mini batches are equal
    # labels = corresponding labels
    nb_samples = len(f_names)
    labels = labels_of_files(f_names)
        
    if nb_samples % batch_size:
        nb_comlementary_samples = batch_size - nb_samples % batch_size
        comlementary_samples = np.random.randint(nb_samples, size=(nb_comlementary_samples))
        complementry_f_names = [f_names[i] for i in comlementary_samples]
        complementry_labels = [labels[i] for i in comlementary_samples]
        complementry_arrays = [arrays[i] for i in comlementary_samples]

        f_names = f_names + complementry_f_names
        labels = labels + complementry_labels
        arrays = arrays + complementry_arrays
        
    labels = np.asarray(labels)
    arrays = np.asarray(arrays)

    return f_names, labels #, arrays

## Class for data generators: incremental allocation of memory for an optimal training/inference process
class DataGenerator(tf.keras.utils.Sequence):
    '[Source] https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly ' 
    'Generates data for Keras'
    def __init__(self, list_IDs, parameters):
        self.dim = parameters["dim"]
        self.batch_size = parameters["batch_size"]
        self.list_IDs = list_IDs
        self.num_classes = parameters["num_classes"]
        self.shuffle = parameters["shuffle"]
        self.on_epoch_end()
        self.is_it_CNN2D = parameters["is_it_CNN2D"]
        self.is_it_CNN1D = parameters["is_it_CNN1D"]
        self.is_it_CNN3D = parameters["is_it_CNN3D"]
        self.VAVp = parameters["VAVp"]
        self.image = parameters["image"]
        self.image3D = parameters["image3D"]
        self.tensorize = parameters["tensorize"]
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            if self.VAVp: # self.is_it_CNN1D:
            # Reshape x_train, x_val, x_test in 4D tensor if CNN 2D
                a, b = read_pt(ID, self.VAVp, self.image, self.image3D, self.tensorize)
                a = np.expand_dims(a, axis=1)
                X[i],y[i] = a, b
            else:
                X[i],y[i] = read_pt(ID, self.VAVp, self.image, self.image3D, self.tensorize)

        if self.is_it_CNN2D:
            # Reshape x_train, x_val, x_test in 4D tensor if CNN 2D
            X = np.expand_dims(X, axis=3)

        return X, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
