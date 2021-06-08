## Read Data
## add libraries
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
random.seed(42)

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

# Nice link: https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://stackoverflow.com/questions/48709839/stopiteration-generator-output-nextoutput-generator
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence?fbclid=IwAR22LzKTJQ_tOrrXuIgzf1ZiqdsKgeRjL_UG6ruSohh53Bm4rWB-3f7xy0I
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/utils/data_utils.py#L397-L472

def get_f_names():
    pt_path = os.path.join('DL_STD_classif','data')
    f_names = os.listdir(pt_path)

    return f_names

def tensorize_egm_matrix(egm_matrix):
    # egm_tensor = np.reshape(egm_matrix, (1,2500, 12), order= 'C' and 'F') 
    egm_tensor = np.empty([1,egm_matrix.shape[1], egm_matrix.shape[0]])
    for l in range(egm_matrix.shape[0]):
        egm_tensor[0,:, l] = egm_matrix[l,:]
    return egm_tensor

def read_pt(pt_name, VAVp, image, image3D, tensorize):
    " pt_name = f_names[pt] " 
    # print(pt_name)
    pt_name_nopath = pt_name
    # pt_path = os.path.join('DL_STD_classif','data')
    # pt_name = os.path.join(pt_path,pt_name)
    # with open(pt_name, 'rb') as ff:
    #     egm_pat_pt_load = pl.load(ff)
    #     label_STD_pat_pat_load = pl.load(ff)
    if image:
        img_path = os.path.join('DL_STD_classif','Bin_EGM_plots1')
        img_name = os.path.join(img_path, pt_name_nopath.split(".")[0]+'.png')
        egm_pat_pt_load = plt.imread(img_name)

        labels_path = os.path.join('DL_STD_classif','labels')
        labels_name = os.path.join(labels_path,pt_name)
        with open(labels_name, 'rb') as ff:
            label_STD_pat_pat_load = pl.load(ff)
    elif image3D:
        img_path = os.path.join('DL_STD_classif','EGM_plots_3slices')
        # img_name = os.path.join(img_path, pt_name_nopath.split(".")[0]+'.dat')
        # # egm_pat_pt_load = plt.imread(img_name)
        # with open(img_name, "rb") as f:
        #     egm_pat_pt_load = pl.load(f) # egm_pat_pt_load_slice

        img_name = os.path.join(img_path, pt_name_nopath.split(".")[0]+'.npy')
        egm_pat_pt_load = np.load(img_name) # egm_pat_pt_load_slice

        labels_path = os.path.join('DL_STD_classif','labels')
        labels_name = os.path.join(labels_path,pt_name)
        with open(labels_name, 'rb') as ff:
            label_STD_pat_pat_load = pl.load(ff)
    else:
        pt_path = os.path.join('DL_STD_classif','data')
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

# def labels_of_files(f_names):
#     labels = []
#     for pt_name in f_names:
#         pt_path = os.path.join('DL_STD_classif','data')
#         pt_name = os.path.join(pt_path,pt_name)
#         with open(pt_name, 'rb') as ff:
#             egm_pat_pt_load = pl.load(ff)
#             label_STD_pat_pat_load = pl.load(ff)            
#         labels.append(label_STD_pat_pat_load)
#     return labels, pt_name
    
def labels_of_files(f_names):
    labels = []
    for pt_name in f_names:
        labels_path = os.path.join('DL_STD_classif','labels')
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

## DA - oversampling 
def balance_oversampling_files(x, VAVp, image, image3D, tensorize):
    # X = np.reshape(x,[x.shape[0],-1],order='F')
    y = read_data_for_ros(x, VAVp, image, image3D, tensorize)
    x = np.asarray(x)
    x = np.expand_dims(x, axis=1)
    ros = imbl.over_sampling.RandomOverSampler(random_state=42)
    y = np.squeeze(y)
    X_resampled, y_resampled = ros.fit_resample(x, y)
    # reshape_dims = [X_resampled.shape[0]]+list(x.shape)[1:]
    # return np.reshape(X_resampled, reshape_dims, order='F'),y_resampled
    X_resampled = np.squeeze(X_resampled)
    X_resampled = X_resampled.tolist()
    return X_resampled ,y_resampled


def read_data_for_ros(f_names, VAVp, image, image3D, tensorize):
    # arrays = []
    labels = labels_of_files(f_names)
    labels = np.asarray(labels)
    return labels

# def adjust_batches(f_names, batch_size, VAVp):
""" optimize to avoid reading and storing arrays (x) because not needed anymore """
def adjust_batches_egms(f_names, batch_size, VAVp, image, image3D, tensorize):
    nb_samples = len(f_names)
    labels = labels_of_files(f_names)
        
    if nb_samples % batch_size:
        nb_comlementary_samples = batch_size - nb_samples % batch_size
        comlementary_samples = np.random.randint(nb_samples, size=(nb_comlementary_samples))
        complementry_f_names = [f_names[i] for i in comlementary_samples]
        complementry_labels = [labels[i] for i in comlementary_samples]
        # complementry_arrays = [arrays[i] for i in comlementary_samples]

        f_names = f_names + complementry_f_names
        labels = labels + complementry_labels
        # arrays = arrays + complementry_arrays
        
    labels = np.asarray(labels)
    # arrays = np.asarray(arrays)

    return f_names, labels #, arrays

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
        # int(ceil(len(self.list_IDs) / self.batch_size)) 
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # if index==len(self):
        #     indexes = self.indexes[:self.batch_size]
        # else:
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
