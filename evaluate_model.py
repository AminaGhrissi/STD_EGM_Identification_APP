from sklearn.metrics import classification_report, confusion_matrix  
import pickle as pl
import os
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import roc_auc_score
import tensorflow as tf 
from sklearn.metrics import confusion_matrix

def eval_and_store_gen(model, model_dir, augment, history, elapsed, test_generator):
    " works only if its input data is an interator: test_generator"
    training_acc = history['accuracy']
    val_acc = history['val_accuracy']
    training_AUC = history['auc']
    val_AUC = history['val_auc']

    val_recall = history['val_recall']
    val_precision = history['val_precision']

    # test_scores list corresponds to  model.metrics_names = ['loss', 'auc', 'accuracy']
    test_scores = model.evaluate(test_generator, verbose=1)

    with open(os.path.join(model_dir,'MLR_save_cm_time.txt'),'a') as f_conf:
        f_conf.write('augment or not with ros = ' + str(augment)+'\n')
        f_conf.write('elaspsed time = ' + str(elapsed)+'\n')
        f_conf.write('test_scores = '+'\n')
        f_conf.write(str(model.metrics_names) +' = '+ str(test_scores)+'\n')

        f_conf.write('train_acc = ' + str(training_acc[-1])+'\n')
        f_conf.write('train_AUC = ' + str(training_AUC[-1])+'\n')
        f_conf.write('train_TPR = ' + str(training_recall[-1])+'\n')
        f_conf.write('train_TNR = ' + str(training_tn[-1]/(training_tn[-1] + training_fp[-1]))+'\n')
        f_conf.write('train_Precision = ' + str(training_precision[-1])+'\n')
        f_conf.write('train_NPV = ' + str(training_tn[-1] / (training_tn[-1] + training_fn[-1]) )+'\n')
        f_conf.write('\n')

        f_conf.write('val_acc = ' + str(val_acc[-1])+'\n')
        f_conf.write('val_AUC = ' + str(val_AUC[-1])+'\n')
        f_conf.write('val_TPR = ' + str(val_recall[-1])+'\n')
        f_conf.write('val_TNR = ' + str(val_tn[-1]/(val_tn[-1] + val_fp[-1]))+'\n')
        f_conf.write('val_Precision = ' + str(val_precision[-1])+'\n')
        f_conf.write('val_NPV = ' + str(val_tn[-1] / (val_tn[-1] + val_fn[-1]) )+'\n')
        f_conf.write('\n')

        f_conf.write('test_acc = ' + str(test_scores[5])+'\n')
        f_conf.write('test_AUC = ' + str(test_scores[8])+'\n')
        f_conf.write('test_TPR = ' + str(test_scores[7])+'\n')
        f_conf.write('test_TNR = ' + str(test_scores[3]/ (test_scores[3]+test_scores[2])) +'\n')
        f_conf.write('test_Precision (PPV) = ' + str(test_scores[6])+'\n')
        f_conf.write('test_NPV = ' + str(test_scores[3]/ (test_scores[3]+test_scores[4])) +'\n')

        f_conf.write('\n')
        f_conf.write('________________________________________________________________'+'\n'+'\n')
        f_conf.write(str(model.summary())+'\n')
    print('Done.')

    PIK = os.path.join(model_dir, 'history.dat')
    with open(PIK, 'wb') as f:
        pl.dump(history, f)  
    """
    import pickle as pl
    with open(PIK, "rb") as f:
        history_loaded = pl.load(f)
    # acc =  history_loaded['accuracy']
    """

########################################################################################################
def eval_and_store(model, model_dir, augment, history, elapsed, train_generator, y_train, val_generator, y_val, test_generator, y_test):
    " works only if its input data is an interator: test_generator"
    training_acc = history['accuracy']
    val_acc = history['val_accuracy']
    
    y_train_pred = model.predict_generator(train_generator)
    y_val_pred = model.predict_generator(val_generator)
    y_test_pred = model.predict_generator(test_generator)

    y_train_categ = tf.keras.utils.to_categorical(y_train, 2)
    y_val_categ = tf.keras.utils.to_categorical(y_val, 2)
    y_test_categ = tf.keras.utils.to_categorical(y_test, 2)

    train_AUC = roc_auc_score(y_train_categ, y_train_pred)
    val_AUC = roc_auc_score(y_val_categ, y_val_pred)
    test_AUC = roc_auc_score(y_test_categ, y_test_pred)

    y_train_pred = np.argmax(y_train_pred, axis = 1)
    y_val_pred = np.argmax(y_val_pred, axis = 1)
    y_test_pred = np.argmax(y_test_pred, axis = 1)

    cm_train = confusion_matrix(y_train,y_train_pred)
    cm_val = confusion_matrix(y_val,y_val_pred)
    cm_test = confusion_matrix(y_test,y_test_pred)

    cmn_train = confusion_matrix(y_train,y_train_pred, normalize = 'true')
    cmn_val = confusion_matrix(y_val,y_val_pred, normalize = 'true')
    cmn_test = confusion_matrix(y_test,y_test_pred, normalize = 'true')

    PPV_train, NPV_train = PPV_NTV(cm_train)
    PPV_val, NPV_val = PPV_NTV(cm_val)
    PPV_test, NPV_test = PPV_NTV(cm_test)

    F1_train = 2*PPV_train* cmn_train[0,0]/(PPV_train + cmn_train[0,0])
    F1_val = 2*PPV_val* cmn_val[0,0]/(PPV_val + cmn_val[0,0])
    F1_test = 2*PPV_test* cmn_test[0,0]/(PPV_test + cmn_test[0,0])

    # test_scores list corresponds to  model.metrics_names = ['loss', 'auc', 'accuracy']
    test_scores = model.evaluate(test_generator, verbose=1)

    with open(os.path.join(model_dir,'performance.txt'),'a') as f_conf:
        f_conf.write('augment or not with ros = ' + str(augment)+'\n')
        f_conf.write('elaspsed time = ' + str(elapsed)+'\n')
        f_conf.write('test_scores = '+'\n')
        f_conf.write(str(model.metrics_names) +' = '+ str(test_scores)+'\n')

        f_conf.write('train_acc = ' + str(training_acc[-1])+'\n')
        f_conf.write('train_AUC = ' + str(train_AUC)+'\n')
        f_conf.write('train_TPR = ' + str(cmn_train[0,0])+'\n')
        f_conf.write('train_TNR = ' + str(cmn_train[1,1])+'\n')
        f_conf.write('train_PPV= ' + str(PPV_train)+'\n')
        f_conf.write('train_NPV = ' + str(NPV_train)+'\n')
        f_conf.write('F1_train = ' + str(F1_train)+'\n')
        f_conf.write('cm = ' + str(cm_train)+'\n')
        f_conf.write('\n')

        f_conf.write('val_acc = ' + str(val_acc[-1])+'\n')
        f_conf.write('val_AUC = ' + str(val_AUC)+'\n')
        f_conf.write('val_TPR = ' + str(cmn_val[0,0])+'\n')
        f_conf.write('val_TNR = ' + str(cmn_val[1,1])+'\n')
        f_conf.write('val_PPV = ' + str(PPV_val)+'\n')
        f_conf.write('val_NPV = ' + str(NPV_val)+'\n')
        f_conf.write('F1_val= ' + str(F1_val)+'\n')
        f_conf.write('cm = ' + str(cm_val)+'\n')
        f_conf.write('\n')

        f_conf.write('test_acc = ' + str(test_scores[6])+'\n')
        f_conf.write('test_AUC = ' + str(test_AUC)+'\n')
        f_conf.write('test_TPR = ' + str(cmn_test[0,0])+'\n')
        f_conf.write('test_TNR = ' + str(cmn_test[1,1]) +'\n')
        f_conf.write('test_PPV = ' + str(PPV_test)+'\n')
        f_conf.write('test_NPV = ' + str(NPV_test) +'\n')
        f_conf.write('F1_test = ' + str(F1_test)+'\n')
        f_conf.write('cm = ' + str(cm_test)+'\n')
        f_conf.write('\n')

        f_conf.write('________________________________________________________________'+'\n'+'\n')
        f_conf.write(str(model.summary())+'\n')
    print('Done.')

    PIK = os.path.join(model_dir, 'history.dat')
    with open(PIK, 'wb') as f:
        pl.dump(history, f)  
    """
    import pickle as pl
    with open(PIK, "rb") as f:
        history_loaded = pl.load(f)
    # acc =  history_loaded['accuracy']
    """
    return test_scores[6], test_AUC, cmn_test[0,0], cmn_test[1,1], PPV_test, NPV_test, F1_test
###########################################################################################################
def PPV_NTV(cm):
    PPV = cm[0,0] / (cm[0,0]+cm[1,0])
    NPV = cm[1,1] / (cm[1,1]+cm[0,1])

    return PPV, NPV 

def plots(model_dir, history):
    
    plot_dir = os.path.join(model_dir,'figures')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    training_acc = history['accuracy']
    training_loss = history['loss']
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']
    training_AUC = history['auc']
    val_AUC = history['val_auc']

    epoch_count = range(1, len(training_loss)+1) # create count of the number of epochs

    # visualize loss history
    fig_handle01 = plt.figure(1)
    plt.plot(epoch_count, training_loss, 'r-',label='Training Loss')
    plt.plot(epoch_count, val_loss, 'b-',label='Validation Loss')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = ''#'Normal Intialization'
    plt.title(title,y=1.08)

    # save figures 
    # pl.dump(fig_handle01,file(plot_dir,'w'))
    plt.savefig(os.path.join(plot_dir, 'loss.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(plot_dir, 'loss.eps'), bbox_inches='tight')
    plt.clf()

    # visualize accuracy history
    fig_handle02 = plt.figure(2)
    plt.ylim(0, 1)
    plt.plot(epoch_count, training_acc, 'r-',label='Training Accuracy')
    plt.plot(epoch_count, val_acc, 'b-',label='Validation Accuracy')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title,y=1.08)
    axes = plt.gca()
    axes.set_ylim([0,1])

    # save figures 
    # pl.dump(fig_handle01,file(plot_dir,'w'))
    plt.savefig(os.path.join(plot_dir , 'acc.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(plot_dir,'acc.eps'), bbox_inches='tight')
    plt.clf()

    # visualize AUC history
    fig_handle03 = plt.figure(3)
    plt.ylim(0, 1)
    plt.plot(epoch_count, training_acc, 'r-',label='Training AUC')
    plt.plot(epoch_count, val_acc, 'b-',label='Validation AUC')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title(title,y=1.08)
    axes = plt.gca()
    axes.set_ylim([0,1])

    # save figures 
    # pl.dump(fig_handle01,file(plot_dir,'w'))
    plt.savefig(os.path.join(plot_dir , 'AUC.png'), bbox_inches='tight')
    # plt.savefig(os.path.join(plot_dir,'AUC.eps'), bbox_inches='tight')
    plt.clf()


def eval_and_store_svm(model, model_dir, augment, elapsed, x_t_train, y_train, x_t_test, y_test, r):
    y_train_pred = model.predict(x_t_train)
    y_test_pred = model.predict(x_t_test)

    train_AUC = roc_auc_score(y_train, y_train_pred)
    test_AUC = roc_auc_score(y_test, y_test_pred)

    cm_train = confusion_matrix(y_train,y_train_pred)
    cm_test = confusion_matrix(y_test,y_test_pred)

    cmn_train = confusion_matrix(y_train,y_train_pred, normalize = 'true')
    cmn_test = confusion_matrix(y_test,y_test_pred, normalize = 'true')

    training_acc = (cm_train[0,0]+cm_train[1,1])/y_train.shape[0] 
    test_acc = (cm_test[0,0]+cm_test[1,1])/y_test.shape[0] 

    PPV_train, NPV_train = PPV_NTV(cm_train)
    PPV_test, NPV_test = PPV_NTV(cm_test)

    F1_train = 2*PPV_train* cmn_train[0,0]/(PPV_train + cmn_train[0,0])
    F1_test = 2*PPV_test* cmn_test[0,0]/(PPV_test + cmn_test[0,0])

    with open(os.path.join(model_dir,'performance.txt'),'a') as f_conf:
        f_conf.write('augment or not with ros = ' + str(augment)+'\n')
        f_conf.write('elaspsed time = ' + str(elapsed)+'\n')
        f_conf.write('Nb of PCs (r) = ' + str(r)+'\n')

        f_conf.write('train_acc = ' + str(training_acc)+'\n')
        f_conf.write('train_AUC = ' + str(train_AUC)+'\n')
        f_conf.write('train_TPR = ' + str(cmn_train[0,0])+'\n')
        f_conf.write('train_TNR = ' + str(cmn_train[1,1])+'\n')
        f_conf.write('train_PPV= ' + str(PPV_train)+'\n')
        f_conf.write('train_NPV = ' + str(NPV_train)+'\n')
        f_conf.write('F1_train = ' + str(F1_train)+'\n')
        f_conf.write('cm = ' + str(cm_train)+'\n')
        f_conf.write('\n')

        f_conf.write('test_acc = ' + str(test_acc)+'\n')
        f_conf.write('test_AUC = ' + str(test_AUC)+'\n')
        f_conf.write('test_TPR = ' + str(cmn_test[0,0])+'\n')
        f_conf.write('test_TNR = ' + str(cmn_test[1,1]) +'\n')
        f_conf.write('test_PPV = ' + str(PPV_test)+'\n')
        f_conf.write('test_NPV = ' + str(NPV_test) +'\n')
        f_conf.write('F1_test = ' + str(F1_test)+'\n')
        f_conf.write('cm = ' + str(cm_test)+'\n')
        f_conf.write('\n')

        f_conf.write('________________________________________________________________'+'\n'+'\n')
    print('Done.')

    return test_acc, test_AUC, cmn_test[0,0], cmn_test[1,1], PPV_test, NPV_test, F1_test


