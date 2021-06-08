# d = x_train[0,:,:]

# d = np.transpose(d)

# for l in range(12):
#     plt.plot(d[l,:]+1*l, 'r') # plotting t, a separately 

# plt.savefig(os.path.join(model_dir, 'subplot_EGMs.png'), bbox_inches='tight')

# resize image and force a new shape
# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
# load image as pixel array
image = image.imread('DL_STD_classif/classif/LSTM_VAVp_generator_categ_VAVp_ros/subplot_EGMs.png')
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()