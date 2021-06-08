import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, concatenate, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv1D, AveragePooling1D
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16

def lenet(num_classes, input_shape):
    conv1_size = (3,3) # (3,4) # (300,4) # (200,4) # 
    egm_inputs = Input(shape=input_shape)
    egm_feature = Conv2D(filters=32, kernel_size=conv1_size, activation='relu')(egm_inputs)
    egm_feature = AveragePooling2D()(egm_feature)
    egm_feature = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(egm_feature)
    egm_feature = AveragePooling2D()(egm_feature)
    egm_feature = Flatten()(egm_feature)
    egm_feature = Dense(units=128, activation='relu')(egm_feature)
    egm_feature = Dense(units=64, activation='relu')(egm_feature)
    egm_feature = Dense(units=num_classes, activation = 'softmax')(egm_feature)
    # egm_feature = Dense(units=1, activation = 'softmax')(egm_feature)
    egm_model = Model(inputs=egm_inputs, outputs=egm_feature)

    egm_model.summary()
    return egm_model

def lenet_drop(num_classes, input_shape):
    conv1_size = (3,4) # (200,4) # (3,4) # (300,4)
    egm_inputs = Input(shape=input_shape)
    egm_feature = Conv2D(filters=32, kernel_size=conv1_size, activation='relu')(egm_inputs)
    egm_feature = Dropout(0.2)(egm_feature)
    egm_feature = AveragePooling2D()(egm_feature)
    egm_feature = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(egm_feature)
    egm_feature = Dropout(0.2)(egm_feature)
    egm_feature = AveragePooling2D()(egm_feature)
    egm_feature = Flatten()(egm_feature)
    egm_feature = Dense(units=128, activation='relu')(egm_feature)
    egm_feature = Dense(units=64, activation='relu')(egm_feature)
    egm_feature = Dropout(0.5)(egm_feature)
    egm_feature = Dense(units=num_classes, activation = 'softmax')(egm_feature)
    # egm_feature = Dense(units=1, activation = 'softmax')(egm_feature)
    egm_model = Model(inputs=egm_inputs, outputs=egm_feature)

    egm_model.summary()
    return egm_model

def mlr(num_classes, input_shape):
    egm_inputs = Input(shape=input_shape)
    egm_feature = Flatten()(egm_inputs)
    egm_feature = Dense(num_classes, activation='softmax')(egm_feature)
    # egm_feature = Dense(1, activation='softmax')(egm_feature)
    egm_model = Model(inputs=egm_inputs, outputs=egm_feature)

    egm_model.summary()
    return egm_model 

def cnn_1D(num_classes, input_shape):
    egm_inputs = Input(shape=input_shape)
    egm_feature = Conv1D(32,kernel_size=3)(egm_inputs)
    egm_feature = AveragePooling1D()(egm_feature)
    egm_feature = Flatten(name = 'flatten_1')(egm_feature)
    egm_feature = Dense(num_classes, activation='softmax')(egm_feature)
    egm_model = Model(inputs=egm_inputs, outputs=egm_feature)
    return egm_model 

def LSTM_VAVp(num_classes, input_shape):
    egm_inputs = Input(shape=input_shape)
    egm_feature = LSTM(32, activation='softmax')(egm_inputs)
    egm_feature = Dense(units=16, activation = 'softmax')(egm_feature)
    egm_feature = Dense(units=num_classes, activation = 'softmax')(egm_feature)
    egm_model = Model(inputs=egm_inputs, outputs=egm_feature)
    return egm_model 

def VGG16_EGM(num_classes, input_shape):
    egm_inputs = VGG16(include_top=False, input_shape=input_shape)
    # add new classifier layers
    egm_feature = Flatten(name='flat')(egm_inputs.layers[-1].output)
    egm_feature = Dense(1024, activation='relu', name='dense1')(egm_feature)
    egm_feature = Dense(128, activation='relu', name='dense2')(egm_feature)
    egm_feature = Dense(2, activation='softmax', name='pred')(egm_feature)
    # define new model
    egm_model = Model(inputs=egm_inputs.inputs, outputs=egm_feature)

    l=0
    for layer in egm_model.layers[:-4]:
        l+=1
        print(layer.name)
        layer.trainable = False
    print(l)

    return egm_model 
