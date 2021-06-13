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
# To generate 2D binary images (plots of the 10 EGM channels)
# 1) run 'save_matrices_as_images.py' 
# 2) run 'save_PNGimages_as_BINimages.py'
# To generate 3D binary images 
# 3) run 'save_matrices_as_images_3slices_npy_npy.py' after 1) and 2)
#################################################################################

import matplotlib.pyplot as plt
import os
import pickle as pl
from PIL import Image
from data_io import get_f_names
import numpy as np

def read_pt_saveas_img_BIN(pt_name, max_lead, slice):
    " pt_name = f_names[pt] " 
    pt_name_nopath = pt_name
    pt_path = os.path.join('data')
    pt_name = os.path.join(pt_path,pt_name)
    with open(pt_name, 'rb') as ff:
        egm_pat_pt_load = pl.load(ff)

    # egm_pat_pt_load = np.transpose(egm_pat_pt_load)
    egm_pat_pt_load = egm_pat_pt_load.astype('float32')
    for l in range(slice): # (egm_pat_pt_load.shape[0]):
        plt.plot(egm_pat_pt_load[10-l,:]+1.1*max_lead*l, 'k')

    for l in range(10-slice): # (egm_pat_pt_load.shape[0]):
        plt.plot(egm_pat_pt_load[l,:]+1.1*max_lead*(l+slice), 'k') # plotting t, a separately 

    img_path = os.path.join('EGM_plots'+str(slice+1))
    if not os.path.exists(img_path): # Create target Directory if doesn't exist
        os.mkdir(img_path)

    plt.axis('off')
    plt.savefig(os.path.join(img_path, pt_name_nopath.split(".")[0]+'.png'), bbox_inches='tight')
    plt.close('all') 
    ##

    Bin_img_path = os.path.join('Bin_EGM_plots'+str(slice+1))
    if not os.path.exists(Bin_img_path): # Create target Directory if doesn't exist
        os.mkdir(Bin_img_path)

    img_name = os.path.join(img_path, pt_name_nopath.split(".")[0]+'.png')
    Bin_img_name = os.path.join(Bin_img_path, pt_name_nopath.split(".")[0]+'.png')

    egm_pat_pt_load = Image.open(img_name)
    gray = egm_pat_pt_load.convert('L')
    bw = gray.point(lambda x: 0 if x<128 else 255, '1')
    bw.save(Bin_img_name)
    plt.close('all') 

    slice_egm_pat_pt_load = plt.imread(Bin_img_name)

    return slice_egm_pat_pt_load

def get_max_read_pt(pt_name):
    " pt_name = f_names[pt] " 
    pt_path = os.path.join('data')
    pt_name = os.path.join(pt_path,pt_name)
    with open(pt_name, 'rb') as ff:
        egm_pat_pt_load = pl.load(ff)

    egm_pat_pt_load = egm_pat_pt_load.astype('float32')
    m = abs(egm_pat_pt_load).max()
    return m 

f_names = get_f_names()

# max_lead = 0
# slice = 1

img3D_path = img_path = os.path.join('EGM_plots3D')
if not os.path.exists(img3D_path): # Create target Directory if doesn't exist
    os.mkdir(img3D_path)

for pt_name in f_names:
    print(pt_name)
    max_lead = get_max_read_pt(pt_name) 

    slice2_egm_pat_pt_load = read_pt_saveas_img_BIN(pt_name, max_lead, 1)
    slice3_egm_pat_pt_load = read_pt_saveas_img_BIN(pt_name, max_lead, 2)

    img_path1 = os.path.join('Bin_EGM_plots1')
    img_name = os.path.join(img_path1, pt_name.split(".")[0]+'.png')
    slice1_egm_pat_pt_load = plt.imread(img_name)

    egm_pat_pt_load_slice = np.empty(shape=(389, 515, 3))
    egm_pat_pt_load_slice[:,:,0] = slice1_egm_pat_pt_load
    egm_pat_pt_load_slice[:,:,1] = slice2_egm_pat_pt_load
    egm_pat_pt_load_slice[:,:,2] = slice3_egm_pat_pt_load

    label_path = os.path.join('data')
    with open(os.path.join(label_path,pt_name), 'rb') as ff:
        egm_matrix = pl.load(ff)
        lab = pl.load(ff)

    if not os.path.exists(os.path.join('EGM_plots_3slices')): # Create target Directory if doesn't exist
        os.mkdir(os.path.join('EGM_plots_3slices'))

    # PIK = os.path.join('EGM_plots_3slices', pt_name.split(".")[0]+'.dat')
    # with open(PIK, 'wb') as f:
    #     pl.dump(egm_pat_pt_load_slice, f) 
    #     pl.dump(lab, f)  
    
    PIK = os.path.join('EGM_plots_3slices', pt_name.split(".")[0]+'.npy')
    np.save(PIK, egm_pat_pt_load_slice)

    """
    with open(PIK, "rb") as f:
        egm_pat_pt_load_slice = pl.load(f)
        lab = pl.load(f)

    for s in range(3):
        sl = egm_pat_pt_load_slice[:,:,s]

        np.save(os.path.join(pt_name.split(".")[0]+'_s'+str(s)+'.npy'), sl)
        sl_loaded = np.load(os.path.join(pt_name.split(".")[0]+'_s'+str(s)+'.npy'))


        plt.imshow(sl)
        plt.savefig(os.path.join(pt_name.split(".")[0]+'_s'+str(s)+'.png'), bbox_inches='tight')
        plt.close('all')
    np.unique(sl)
    """

t = 1+2
print(t)

# https://towardsdatascience.com/why-you-should-start-using-npy-file-more-often-df2a13cc0161



