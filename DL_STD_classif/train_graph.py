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
from tensorflow.keras.optimizers import Adam
import os
import time 

from def_models import mlr, lenet, lenet_drop, cnn_1D, LSTM_VAVp, VGG16_EGM

## Class for training with generators
class STD_classifier:
    def __init__(self, parameters,input_shape, model_dir):
        self.parameters = parameters
        self.input_shape = input_shape
        self.model_dir = model_dir

    def call_functions(self):
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.model_dir,'training.csv'))
        early = tf.keras.callbacks.EarlyStopping(monitor=self.parameters["monitor"], patience=self.parameters["patience"],mode=self.parameters["mode"])
        # checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, 'model.{epoch:02d}-{'+monitor+':.2f}.hdf5'), monitor=monitor)
        # Metric(validation_generator, wandb),
        calls = [csv_logger, early]
        return calls

    def get_architecture(self):
        if self.parameters["architecture"] == "mlr":
            return mlr
        elif self.parameters["architecture"] == "lenet":
            return lenet
        elif self.parameters["architecture"] == "lenet_drop":
            return lenet_drop
        elif self.parameters["architecture"] == "cnn_1D":    
            return cnn_1D
        elif self.parameters["architecture"] == "LSTM_VAVp":
            return LSTM_VAVp
        elif self.parameters["architecture"] == "VGG16_EGM":
            return VGG16_EGM

    def trainer_gen(self, architecture, calls, train_generator, val_data): #, wandb):
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
        model = architecture(self.parameters["num_classes"], self.input_shape)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(), # BinaryCrossentropy(), #'categorical_crossentropy', # tf.keras.losses.CategoricalCrossentropy(), #
            optimizer=Adam(self.parameters["learning_rate"]),
            metrics=metrics_to_compute,
        )
        t = time.time() 
        # https://stackoverflow.com/questions/55531427/how-to-define-max-queue-size-workers-and-use-multiprocessing-in-keras-fit-gener
        history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=self.parameters["steps_per_epoch"],
            epochs=self.parameters["epochs"],
            verbose=1,
            validation_data=val_data,
            validation_steps=self.parameters["validation_steps"], 
            workers=-1, # 6,
            callbacks=calls, 
            use_multiprocessing = True,
        )
        elapsed = time.time() - t 
        return model, history, elapsed

    def trainer(self, architecture, calls, x_train, y_train, x_val, y_val): #, wandb):
        metrics_to_compute = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        ]
        model = architecture(self.parameters["num_classes"], self.input_shape)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(self.parameters["learning_rate"]),
            metrics=metrics_to_compute,
        )
        t = time.time() 
        history = model.fit(x_train, y_train,
          batch_size=self.parameters["batch_size"],
          epochs=self.parameters["epochs"],
          verbose=1,
          validation_data=(x_val, y_val), callbacks=calls)
        elapsed = time.time() - t 
        return model, history, elapsed