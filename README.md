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

# DL_STD_classif
Classification of 955 real 10-channel EGM recordings into STD vs. non-STD groups using machine learning algorithms.
____________________________________________________________________________________________

The main script is called 'main_classgen_5fold.py'
Input 1: EGM recordings (12 channels of 2.5 s sampled at 1 kHz) stored in 'data' folder, each sample in a .pickle file to facilitate the use of data generators.
Each EGM sample has 12 channels (not 10), resulting from the duplication of the first two channels at the end of the recording in order to mimic the circularity of the Pentaray catheter branches. 

Input 2: EGM labels, binary (0 for non-STD, 1 for STD).

Classification performance over kfold cross-validation stored in 'classification' folder.
____________________________________________________________________________________________

The main allows to use different models, a model is a combination of a 'data format' and a 'classification algorithm'.

Four data formats are proposed: 
1- matrices (10x2500)
2- 2D images : a plot of the 10 EGM channels on the same figure one under the other
3- 3D image : a 3D image of depth 3, each slice of it is a 2D image a plot of the 10 EGM channels on the same figure one under the other, a circular rotation of the EGM channels 
is performed from one slice to the next one
4- VAVp: a 1D time series called Maximal Voltage Absolute Values at any of the PentaRay bipoles. VAVp computed from the EGM matrix, detailed in [A. Ghrissi, https://ieeexplore.ieee.org/abstract/document/9287681] 

The classification algorithms used are: 
- lenet: a shallow convolutional neural network (CNN) inspired from LeNet5 architecture
         the filter size of the first convolutional (conv) layer is set to (4 x alpha AFLC), AFCL is AF cycle length (typical = 200 ms)
 	 the filter size is chosen to be (4x300, 4x200, 4x3)

- mlr: a neural network (NN) modeled as a multivariate logistic regression
- cnn_1D: a 1D shallow CNN
- pca_svm: principal component analysis (PCA) followed by support vector machine (SVM) 
- VGG16_EGM: transfer learning of VGG16 CNN architecture

___________________________________________________________________________________________

Depending on the classification model (data format + algorithm), the value of the data dimensions is required, it is called "dim"
The settings of a model are specified by the user in the dictionary "parameters" as follows

- matrix format, dim = (2500,12)
- VAVp + pca_svm, dim = (2500, 1)
- VAVp + cnn_1D, dim = (2500, 1)
- 3D image + VGG16, dim = (389, 515, 3)
- 2D image + mlr, dim = (389, 515)
- 2D image + lenet, dim = (389, 515)

_______________________________________________________________________________________________

To generate 2D binary images (plots of the 10 EGM channels)
1) run 'save_matrices_as_images.py' 
2) run 'save_PNGimages_as_BINimages.py'

To generate 3D binary images 
3) run 'save_matrices_as_images_3slices_npy_npy.py' after running 1) and 2) respectively 


