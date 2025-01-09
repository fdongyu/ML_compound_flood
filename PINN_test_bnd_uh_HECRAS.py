import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from datetime import datetime
import time
from pyDOE import lhs
import tensorflow as tf
import pickle

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/qfs/people/feng779/PYTHON/utils')

from utils_ML import readhdf, readslf
from SVE_module_dynamic_uh_mff_ts_l2_new import SVE

import pdb


if __name__ == "__main__":

    """
    step one: read HEC-RAS output
    """
    starttime = datetime(2011, 8, 26)
    endtime = datetime(2011, 9, 5)

    hdf_filename = 'files/DR_1D_5cells.p01.hdf'
    
    x_values, elev, slope, time_model, water_surface, water_depth, u, v = readhdf(hdf_filename, starttime, endtime)


    b = 2000 # m, channel width

    Nt = water_depth.shape[0]
    Nx = water_depth.shape[1]


    Nt_train = water_depth.shape[0] ## consider all time steps
    layers = [2] + 5*[1*64] + [2]


    t = np.arange(Nt_train)[:,None]
    x = x_values[:,None]
    X, T = np.meshgrid(x,t)
    u_exact = v[:Nt_train,:] 
    h_exact = water_depth[:Nt_train,:]


    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = u_exact.flatten()[:,None]
    h_star = h_exact.flatten()[:,None]
    
    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    ##  start and end time steps
    tsteps = [0] + [Nt-1]

    for i, tstep in enumerate(tsteps):
        xx1_ = np.hstack((X[tstep:tstep+1,:].T, T[tstep:tstep+1,:].T))
        hh1_ = h_exact[tstep:tstep+1,:].T
        uu1_ = u_exact[tstep:tstep+1,:].T
        if i == 0:
            xx1 = xx1_
            hh1 = hh1_
            uu1 = uu1_
        else:
            xx1 = np.vstack((xx1, xx1_))
            hh1 = np.vstack((hh1, hh1_))
            uu1 = np.vstack((uu1, uu1_))

    xx2 = np.hstack((X[:,0:1], T[:,0:1]))   ## dnstrm BC
    uu2 = u_exact[:,0:1]
    hh2 = h_exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))   ## upstrm BC
    uu3 = u_exact[:,-1:]
    hh3 = h_exact[:,-1:]

    X_u_IC = xx1
    X_h_IC = xx1
    u_IC = uu1
    h_IC = hh1
    X_u_BC = np.vstack([xx2, xx3])
    X_h_BC = np.vstack([xx2, xx3])
    u_BC = np.vstack([uu2, uu3])
    h_BC = np.vstack([hh2, hh3])


    useObs = True
    ## coarser model domain

    ind_obs = np.arange(0,Nx,25)[1:]
    t_obs_u = np.array([])
    x_obs_u = np.array([])
    u_obs = np.array([])
    for iobs in ind_obs:
        t_obs_u = np.append( t_obs_u, t.flatten() )
        x_obs_u = np.append( x_obs_u, np.ones(Nt_train)*x[iobs] )
        u_obs = np.append( u_obs, u_exact[:Nt_train, iobs] )
    X_u_obs = np.vstack([x_obs_u, t_obs_u]).T
    u_obs = u_obs[:,None]

    t_obs_h = np.array([])
    x_obs_h = np.array([])
    h_obs = np.array([])
    for iobs in ind_obs:
        t_obs_h = np.append( t_obs_h, t.flatten() )
        x_obs_h = np.append( x_obs_h, np.ones(Nt_train)*x[iobs] )
        h_obs = np.append( h_obs, h_exact[:Nt_train, iobs] )
    X_h_obs = np.vstack([x_obs_h, t_obs_h]).T
    h_obs = h_obs[:,None]


    X_f_train = X_star
    slope = np.hstack([np.array(slope) for _ in range(Nt_train)])[:,None]


    exist_mode = 0
    if exist_mode == 0:
        # Training
        starttimer = time.time()
        model = SVE(X_u_IC, X_h_IC,
                    X_u_BC, X_h_BC,
                    X_u_obs,X_h_obs,
                    X_f_train,
                    u_IC, h_IC,
                    u_BC, h_BC,
                    u_obs, h_obs,
                    layers,
                    lb, ub, slope, b,
                    X_star, u_star, h_star,
                    useObs=useObs)

        model.train(num_epochs = 30000) ## 5 layers 2000 or 200000

        endtimer = time.time()
        print ('PINN Time elapsed: ', endtimer-starttimer)

        save_path = 'saved_model/PINN_uh.pickle'
        model.save_NN(save_path)
        weight_path = 'saved_model/PINN_uh_weights.out'
        model.save_weight(weight_path)
        wmff_path = 'saved_model/PINN_uh_mff.out'
        model.save_wmff(wmff_path)


#    pdb.set_trace()

