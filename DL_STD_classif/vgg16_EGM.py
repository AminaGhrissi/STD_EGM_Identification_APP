# Link: https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

# Library
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

## Seems to be useful for incremental memory allocation, otherwise TF consumes all memory of the GPU instantaniously
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# # load model
# model = VGG16()
# # summarize the model
# model.summary()
	
# # remove the output layer
# model_nooutput = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# model_nooutput.summary()

# print('loaded')
	
# example of tending the vgg16 model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
# load model without classifier layers
model = VGG16(include_top=False, input_shape=(300, 300, 3))
# add new classifier layers
flat1 = Flatten(name='flat')(model.layers[-1].output)
class1 = Dense(1024, activation='relu', name='dense1')(flat1)
class2 = Dense(128, activation='relu', name='dense2')(class1)
output = Dense(2, activation='softmax', name='pred')(class2)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize
model.summary()

print('reset1')
l=0
for layer in model.layers[:-4]:
    l+=1
    print(layer.name)
    layer.trainable = False

    