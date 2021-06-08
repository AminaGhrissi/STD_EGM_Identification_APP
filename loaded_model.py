from tensorflow.keras.models import load_model
import os

model_dir = os.path.join('DL_STD_classif','classif', 'cnn_1D_generator_categ_VAVp_ros' , 'model.h5')

# load model
model = load_model(model_dir)
model.summary()
print('done')
