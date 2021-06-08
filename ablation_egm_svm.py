import tensorflow.keras as keras
import tensorflow as tf
#from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Conv1D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LambdaCallback, CSVLogger, ModelCheckpoint

from scipy.io import loadmat
import numpy as np 
import pickle as pl
import time 

from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import precision_recall_curve, log_loss, classification_report
from sklearn.metrics import confusion_matrix, fbeta_score, cohen_kappa_score, matthews_corrcoef, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.externals import joblib
import os

#Seems to be useful for incremental memory allocation, otherwise TF consumes all memory of the GPU instantaniously
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

## classification        
num_classes = 2

batch_size = 32
epochs = 500
epoch_count = range(1, epochs+1) # Create count of the number of epochs
loss = 'categorical_crossentropy' # 'mse'# 
optim = keras.optimizers.Adam()
metrics_to_compute = ['acc',tf.keras.metrics.AUC()]

## callbacks
model_dir='data-io/ablation_egms_aug_SVM' #VAVp_CNN'
monitor='val_auc'
patience=10
mode = 'auto'
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'training.csv'))
early = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,mode=mode)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, 'model.{epoch:02d}-{'+monitor+':.2f}.hdf5'), monitor=monitor)
calls = [csv_logger, early, checkpointer]
# calls = [csv_logger, checkpointer]

## Load VAVp data
from scipy.io import loadmat
data = loadmat('data-io/all_pat_train_test_val_aug.mat')
# save('VAVp_timeseries_aug.mat','VAVp_timeseries_train','egm_STD_test','labels_train','ablation_label_all_test','egm_STD_train_aug','ablation_label_all_train')

egm_STD_train_aug_cell = data['egm_STD_train_aug'][0] 
ablation_label_all_train = data['ablation_label_all_train_aug']
ablation_label_all_train = np.transpose(ablation_label_all_train)
ablation_label_all_train = ablation_label_all_train.ravel()

egm_STD_val_test_cell = data['egm_STD_test'][0]
ablation_label_all_val_test = data['ablation_label_all_test']
ablation_label_all_val_test = np.transpose(ablation_label_all_val_test)
ablation_label_all_val_test = ablation_label_all_val_test.ravel()
##
egm_STD_train_aug = np.zeros((egm_STD_train_aug_cell.shape[0],egm_STD_train_aug_cell[0].shape[0],egm_STD_train_aug_cell[0].shape[1]))
nb_train = len(ablation_label_all_train)
for pt in range(nb_train):
    egm_STD_train_aug[pt] = egm_STD_train_aug_cell[pt] # access egm signasl of point pt"""

egm_STD_val_test = np.zeros((egm_STD_val_test_cell.shape[0],egm_STD_val_test_cell[0].shape[0],egm_STD_val_test_cell[0].shape[1]))
nb_val_test = len(ablation_label_all_val_test)
for pt in range(nb_val_test):
    egm_STD_val_test[pt] = egm_STD_val_test_cell[pt] # access egm signasl of point pt"""

egm_STD_val, egm_STD_test, ablation_label_all_val, ablation_label_all_test = train_test_split(egm_STD_val_test, ablation_label_all_val_test, test_size=0.5, random_state=42)

nb_val = egm_STD_val.shape[0]
nb_test = egm_STD_test.shape[0]

##from here we will be classify the images. First we should flatten the imagee (flatten after suffling)
## https://www.kaggle.com/halien/simple-image-classifer-with-svm

egm_STD_train_aug = egm_STD_train_aug.reshape((nb_train,-1)) # x_train.shape = (1710, 30000), 12*2500 = 30000, nb_train = 1710
egm_STD_val = egm_STD_val.reshape((nb_val,-1))
egm_STD_test = egm_STD_test.reshape((nb_test,-1))

## SVM parameters
kernel = 'linear'
C = 1.0

## SVM training 
t = time.time() 
clf = SVC(kernel = 'linear', C = C)

clf.fit(egm_STD_train_aug, ablation_label_all_train)
y_pred_val = clf.predict(egm_STD_val)  

elapsed = time.time() - t 

joblib.dump(clf, model_dir+'svm_linear.pkl') 

"""
# Load the pickle file
from sklearn.externals import joblib
clf_load = joblib.load('!!!.pkl') 
# Check that the loaded model is the same as the original
assert clf.score(x_test, y_test) == clf_load.score(x_test, y_test)
"""
###############

## confusion matrix
y_pred_train = clf.predict(egm_STD_train_aug)
confusion_mat_train = confusion_matrix(ablation_label_all_train,y_pred_train)

confusion_mat_val = confusion_matrix(ablation_label_all_val,y_pred_val)

y_pred_test = clf.predict(egm_STD_test)
confusion_mat_test = confusion_matrix(ablation_label_all_test,y_pred_test)

###############################################################################
## save y_train & y_test --> neeed for recomputing the CM
###############################################################################
PIK = model_dir+'/y_pred_train.dat'
with open(PIK, 'wb') as f:
    pl.dump(y_pred_train, f)

PIK = model_dir+'/y_pred_val.dat'
with open(PIK, 'wb') as f:
    pl.dump(y_pred_val, f) 

PIK = model_dir+'/y_pred_test.dat'
with open(PIK, 'wb') as f:
    pl.dump(y_pred_test, f) 

## results with test dataset 
TPR = confusion_mat_test[0,0]/(confusion_mat_test[0,0]+confusion_mat_test[0,1])
TNR = confusion_mat_test[1,1]/(confusion_mat_test[1,1]+confusion_mat_test[1,0])
precision = confusion_mat_test[0,0]/(confusion_mat_test[0,0]+confusion_mat_test[1,0])
# ROC area 
fpr, tpr, _ = roc_curve(y_pred_train, ablation_label_all_train)
train_auc = auc(fpr, tpr)
fpr, tpr, _ = roc_curve(y_pred_val, ablation_label_all_val)
val_auc = auc(fpr, tpr)
fpr, tpr, _ = roc_curve(y_pred_test, ablation_label_all_test)
test_auc = auc(fpr, tpr)

with open(model_dir+'/save_cm_time.txt','a') as f_conf:
    f_conf.write('confusion_mat_train = ' + str(confusion_mat_train)+'\n')
    f_conf.write('confusion_mat_val = ' + str(confusion_mat_val)+'\n')
    f_conf.write('confusion_mat_test = ' + str(confusion_mat_test)+'\n')
    f_conf.write('elaspsed time = ' + str(elapsed)+'\n')
    f_conf.write('test TPR = ' + str(TPR)+'\n')
    f_conf.write('test TNR = ' + str(TNR)+'\n')
    f_conf.write('test precision = ' + str(precision)+'\n')
    f_conf.write('train auc = ' + str(train_auc)+'\n')
    f_conf.write('val auc = ' + str(val_auc)+'\n')
    f_conf.write('test auc = ' + str(test_auc)+'\n')

print('hello')
## Read as
"""
import pickle as pl

with open(PIK, "rb") as f:
    y = pl.load(f)
"""