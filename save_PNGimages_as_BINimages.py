import matplotlib.pyplot as plt
import os
import pickle as pl
from PIL import Image
from data_io import get_f_names

def read_pt_saveas_img(pt_name, max_lead):
    " pt_name = f_names[pt] " 
    pt_name_nopath = pt_name
    pt_path = os.path.join('DL_STD_classif','data')
    pt_name = os.path.join(pt_path,pt_name)

    img_path = os.path.join('DL_STD_classif','EGM_plots1')
    Bin_img_path = os.path.join('DL_STD_classif','Bin_EGM_plots1')
    
    img_name = os.path.join(img_path, pt_name_nopath.split(".")[0]+'.png')
    Bin_img_name = os.path.join(Bin_img_path, pt_name_nopath.split(".")[0]+'.png')

    egm_pat_pt_load = Image.open(img_name)
    gray = egm_pat_pt_load.convert('L')
    bw = gray.point(lambda x: 0 if x<128 else 255, '1')
    bw.save(Bin_img_name)

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


