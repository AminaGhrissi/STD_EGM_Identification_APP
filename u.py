##########################################################################################################
""" Adam + def categ acc """
# convert class vectors to binary class matrices

## LR model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

input_shape = list(x_train.shape[1:])

inputs = Input(shape=input_shape)
feature = Flatten()(inputs)
feature = Dense(2, activation='softmax')(feature)
model0_deb1 = Model(inputs=inputs, outputs=feature)

## Compile

monitor = 'val_auc'
patience = 10
mode = 'auto'
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'old_training_deb1.csv'))
early = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,mode=mode)
calls = [csv_logger, early]

loss = 'categorical_crossentropy' # 'mse'# 
optim = keras.optimizers.Adam()
#metrics_to_compute = [tf.keras.metrics.AUC(), 'acc']
metrics_to_compute = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'), # BinaryAccuracy(name='accuracy'), #
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        ]
model0_deb1.compile(loss=loss,
              optimizer=Adam(parameters["learning_rate"]), # optim,
              metrics=metrics_to_compute)
          
## Training
history_deb1 = model0_deb1.fit(x_train, y_train,
          batch_size=parameters["batch_size"],
          epochs=parameters["epochs"],
          verbose=1,
          validation_data=(x_val,y_val), callbacks=calls)

training_acc = history_deb1.history['accuracy']
training_loss = history_deb1.history['loss']
val_acc = history_deb1.history['val_accuracy']
val_loss = history_deb1.history['val_loss']
training_AUC = history_deb1.history['auc']
val_AUC = history_deb1.history['val_auc']

score_MLR0_deb1 = model0_deb1.evaluate(x_test, y_test, verbose=1)
# score_MLR list corresponds to  model.metrics_names = ['loss', 'auc', 'acc']

y_train_pred = model0_deb1.predict(x_train)
y_val_pred = model0_deb1.predict(x_val)
y_test_pred = model0_deb1.predict(x_test)

y_train_pred =np.argmax(y_train_pred, axis = 1)
y_val_pred = np.argmax(y_val_pred, axis = 1)
y_test_pred = np.argmax(y_test_pred, axis = 1)

cm_train0_deb1 = confusion_matrix(y_train_1_column,y_train_pred, normalize = 'true')
cm_val0_deb1 = confusion_matrix(y_val_1_column,y_val_pred, normalize = 'true')
cm_test0_deb1 = confusion_matrix(y_test_1_column,y_test_pred, normalize = 'true')

with open(os.path.join(model_dir,'model0_deb1.txt'),'a') as f_conf:
    f_conf.write(str(model0_deb1.metrics_names) +'\n')
    f_conf.write('test score = '+ str(score_MLR0_deb1)+'\n')
    f_conf.write('cm_train1 = '+ str(cm_train0_deb1)+'\n')
    f_conf.write('cm_val1 = '+ str(cm_val0_deb1)+'\n')
    f_conf.write('cm_test1 = '+ str(cm_test0_deb1)+'\n')