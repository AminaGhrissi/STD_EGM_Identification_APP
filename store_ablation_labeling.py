## add libraries
from scipy.io import loadmat
import numpy as np 
import pickle as pl
import time 
import os
from scipy.spatial.distance import cdist
from scipy.io import loadmat
import pandas as pd

patients_id = [1, 9, 11, 14, 15, 21, 22, 24, 25, 27, 28, 30, 32, 34, 36]

for pat_id in patients_id:

    f_name_pat = 'Ablation_prediction_STD_Python/Ablation_labeling/Ablation_pat_'+str(pat_id)+'.pickle'
    with open(f_name_pat, 'rb') as ff:
        XYZ_penta_pat = pl.load(ff)
        egms_pat = pl.load(ff)
        labels_STD_pat = pl.load(ff)
        i_nearest_pat = pl.load(ff)
        nb_points_pat = pl.load(ff)
        labels_abl_pat = pl.load(ff)
        D_pat = pl.load(ff)
        nb_sessions = pl.load(ff)
        X_Sess_avg = pl.load(ff)
        Y_Sess_avg = pl.load(ff)
        Z_Sess_avg = pl.load(ff)
        X_Sess_std = pl.load(ff)
        Y_Sess_std = pl.load(ff)
        Z_Sess_std = pl.load(ff)
        nb_abl_points_pat = pl.load(ff)
        D_nearest_pat = pl.load(ff)
        nb_points_pentaray_pat = pl.load(ff)

    nb_pts_pat = egms_pat.shape[0]

    for pt in range(nb_pts_pat):
        pt_name = os.path.join('DL_STD_classif','data','pat_'+str(pat_id)+'_pt'+str(pt)+'.pickle')
        egm_pat_pt = egms_pat[pt]
        label_STD_pat_pat = labels_STD_pat[pt][0]

        with open(pt_name,'wb') as f: 
            pl.dump(egm_pat_pt, f)
            pl.dump(label_STD_pat_pat, f) 

print('Done')

# with open(pt_name, 'rb') as ff:
#     egm_pat_pt_load = pl.load(ff)
#     label_STD_pat_pat_load = pl.load(ff)
