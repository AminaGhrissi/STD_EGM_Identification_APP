from scipy.io import loadmat
import numpy as np 
import pickle as pl
import time 

from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import precision_recall_curve, log_loss, classification_report
from sklearn.metrics import confusion_matrix, fbeta_score, cohen_kappa_score, matthews_corrcoef, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import imblearn as imbl

## user's libraries 
from data_io import adjust_batches_egms, adjust_batches, get_f_names, fnames_split_train_val_test, balance_oversampling_files, read_data_for_ros
from evaluate_model import plots, eval_and_store_gen, eval_and_store

## metaparameters
parameters = {
        "is_it_CNN2D": True, # False, # 
        "architecture": "lenet", # "mlr", # 
        "n_folds": 5,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 100,
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
## SVM parameters
kernel = 'linear'
C = 1.0

## directories
model_dir = os.path.join('DL_STD_classif','classif','SVM_'+kernel+'_C'+str(int(C)))
data_dir = os.path.join('DL_STD_classif',)

if parameters["augment"]:
    model_dir += '_ros'    
if not os.path.exists(model_dir): # Create target Directory if doesn't exist
    os.mkdir(model_dir)

## split files_names 
f_names = get_f_names()
train_files, val_files, test_files = fnames_split_train_val_test(f_names)

# oversampling(train file_names)
if parameters["augment"]:
    print('ros training dataset')
    train_files, y_train = balance_oversampling_files(train_files)
    nb_train_aug = len(train_files)
    p = np.random.permutation(nb_train_aug)
    train_files = [train_files[i] for i in p]
    y_train = y_train[p]

train_files, y_train, x_train = adjust_batches_egms(train_files, parameters["batch_size"])
val_files, y_val, x_val = adjust_batches_egms(val_files, parameters["batch_size"])
test_files, y_test, x_test = adjust_batches_egms(test_files, parameters["batch_size"])

nb_train = len(train_files)
nb_val = len(val_files)
nb_test = len(test_files)

##from here we will be classify the images. First we should flatten the imagee (flatten after suffling)
## https://www.kaggle.com/halien/simple-image-classifer-with-svm

x_train = x_train.reshape((nb_train,-1)) # x_train.shape = (1710, 30000), 12*2500 = 30000, nb_train = 1710
x_val = x_val.reshape((nb_val,-1))
x_test = x_test.reshape((nb_test,-1))

## SVM training 
t = time.time() 
clf = SVC(kernel = kernel, C = C,  verbose = True, cache_size=1000, max_iter = 1000)
# https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
# 
# !! The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples. For large datasets consider using sklearn.svm.LinearSVC or sklearn.linear_model.SGDClassifier instead, possibly after a sklearn.kernel_approximation.Nystroem transformer.
clf.fit(x_train, y_train)
y_pred_val = clf.predict(x_val)  

elapsed = time.time() - t 

joblib.dump(clf, os.path.join(model_dir,'svm.pkl')) 

"""
# Load the pickle file
from sklearn.externals import joblib
clf_load = joblib.load('!!!.pkl') 
# Check that the loaded model is the same as the original
assert clf.score(x_test, y_test) == clf_load.score(x_test, y_test)
"""

## confusion matrix
y_pred_train = clf.predict(x_train)
confusion_mat_train = confusion_matrix(y_train, y_pred_train)

confusion_mat_val = confusion_matrix(y_val, y_pred_val)

y_pred_test = clf.predict(x_test)
confusion_mat_test = confusion_matrix(y_test, y_pred_test)

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
PPV = confusion_mat_test[0,0]/(confusion_mat_test[0,0]+confusion_mat_test[1,0])
NPV = confusion_mat_test[1,1]/(confusion_mat_test[1,1]+confusion_mat_test[0,1])
test_acc = (confusion_mat_test[0,0]+confusion_mat_test[1,1])/nb_test

#
train_acc = (confusion_mat_train[0,0]+confusion_mat_train[1,1])/nb_train
val_acc = (confusion_mat_val[0,0]+confusion_mat_val[1,1])/nb_val

# ROC area 
fpr, tpr, _ = roc_curve(y_pred_train, y_train)
train_auc = auc(fpr, tpr)
fpr, tpr, _ = roc_curve(y_pred_val, y_val)
val_auc = auc(fpr, tpr)
fpr, tpr, _ = roc_curve(y_pred_test, y_test)
test_auc = auc(fpr, tpr)

with open(model_dir+'/performance.txt','a') as f_conf:
    f_conf.write('confusion_mat_train = ' + str(confusion_mat_train)+'\n')
    f_conf.write('confusion_mat_val = ' + str(confusion_mat_val)+'\n')
    f_conf.write('confusion_mat_test = ' + str(confusion_mat_test)+'\n')
    f_conf.write('elaspsed time = ' + str(elapsed)+'\n')

    f_conf.write('train acc = ' + str(train_acc)+'\n')
    f_conf.write('train AUC = ' + str(train_auc)+'\n')

    f_conf.write('val acc = ' + str(val_acc)+'\n')
    f_conf.write('val AUC = ' + str(val_auc)+'\n')

    f_conf.write('test acc = ' + str(test_acc)+'\n')
    f_conf.write('test AUC = ' + str(test_auc)+'\n')
    f_conf.write('test TPR = ' + str(TPR)+'\n')
    f_conf.write('test TNR = ' + str(TNR)+'\n')
    f_conf.write('test PPV = ' + str(PPV)+'\n')
    f_conf.write('test NPV = ' + str(NPV)+'\n')

print('Done')
print('Done')
## Read as
"""
import pickle as pl

with open(PIK, "rb") as f:
    y = pl.load(f)
"""