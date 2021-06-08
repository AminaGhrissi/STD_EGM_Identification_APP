import matplotlib.pyplot as plt
import os
import pickle as pl

from data_io import get_f_names

def read_pt_saveas_img(pt_name, max_lead):
    " pt_name = f_names[pt] " 
    pt_name_nopath = pt_name
    pt_path = os.path.join('DL_STD_classif','data')
    pt_name = os.path.join(pt_path,pt_name)
    with open(pt_name, 'rb') as ff:
        egm_pat_pt_load = pl.load(ff)

    # egm_pat_pt_load = np.transpose(egm_pat_pt_load)
    egm_pat_pt_load = egm_pat_pt_load.astype('float32')

    for l in range(10): # (egm_pat_pt_load.shape[0]):
        plt.plot(egm_pat_pt_load[l,:]+1.1*max_lead*l, 'k') # plotting t, a separately 

    img_path = os.path.join('DL_STD_classif','EGM_plots_tif')
    plt.axis('off')
    plt.savefig(os.path.join(img_path, pt_name_nopath.split(".")[0]+'.tif'), bbox_inches='tight')
    plt.close('all') 
    return

def get_max_read_pt(pt_name):
    " pt_name = f_names[pt] " 
    pt_path = os.path.join('DL_STD_classif','data')
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


