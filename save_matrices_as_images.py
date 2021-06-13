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

from data_io import get_f_names

def read_pt_saveas_img(pt_name, max_lead):
    " pt_name = f_names[pt] " 
    pt_name_nopath = pt_name
    pt_path = os.path.join('data')
    pt_name = os.path.join(pt_path,pt_name)
    with open(pt_name, 'rb') as ff:
        egm_pat_pt_load = pl.load(ff)

    # egm_pat_pt_load = np.transpose(egm_pat_pt_load)
    egm_pat_pt_load = egm_pat_pt_load.astype('float32')

    for l in range(10): # (egm_pat_pt_load.shape[0]):
        plt.plot(egm_pat_pt_load[l,:]+1.7*max_lead*l, 'k') # plotting t, a separately 

    img_path = os.path.join('EGM_plots1')
    if not os.path.exists(img_path):
        os.makedirs(img_path) 

    plt.axis('off')
    plt.savefig(os.path.join(img_path, pt_name_nopath.split(".")[0]+'.png'), bbox_inches='tight')
    plt.close('all') 
    return

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
for pt_name in f_names:
    print(pt_name)
    max_lead = get_max_read_pt(pt_name) 
    read_pt_saveas_img(pt_name, max_lead)
    

t = 1+2
print(t)


