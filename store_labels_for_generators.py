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
from data_io import get_f_names
from train_graph import STD_classifier
from evaluate_model import plots, eval_and_store_gen, eval_and_store
from math import ceil
import numpy as np 
from numpy import std, mean 
from operator import itemgetter
from sklearn.model_selection import StratifiedKFold

import pickle as pl

f_names = get_f_names()

if not os.path.exists(os.path.join('DL_STD_classif','labels')): # Create target Directory if doesn't exist
    os.mkdir(os.path.join('DL_STD_classif','labels'))

def labels_of_files(f_names):
    for pt_name in f_names:
        pt_path = os.path.join('DL_STD_classif','data')
        with open(os.path.join(pt_path,pt_name), 'rb') as ff:
            egm_pat_pt_load = pl.load(ff)
            label_STD_pat_pat_load = pl.load(ff)            
        
        PIK = os.path.join('DL_STD_classif','labels', pt_name)
        with open(PIK, 'wb') as f:
            pl.dump(label_STD_pat_pat_load, f) 
        print('stored y:'+pt_name)
        print(str(label_STD_pat_pat_load))

        with open(PIK, 'rb') as f:
            lab = pl.load(f)
        print(str(lab))

    return
labels_of_files(f_names)
print('labels OK')