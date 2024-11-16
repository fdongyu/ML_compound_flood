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

from utils_ML import readslf
from SVE_module_dynamic_uh_mff_ts_l2_new import SVE

import pdb


if __name__ == "__main__":

    """
    step one: read HEC-RAS output
    """
    Nobs = 4

    starttime = datetime(2011, 8, 26)
    endtime = datetime(2011, 9, 5)

    slf_filename = 'files/output_high.slf'
    filename_mesh = 'files/mesh_1D_channel_dx100_update.slf'

    x_values, elev, slope, time_model, water_surface, water_depth, u, v = readslf(slf_filename, filename_mesh, starttime, endtime)

    ## resolution = 400
    x_values = x_values[::2]
    elev = elev[::2]
    slope = slope[::2]
    water_depth = water_depth[:,::2]
    u = u[:,::2]
    v = v[:,::2]

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


    exist_mode = 2
    saved_path = 'saved_model/PINN_uh_Telemac_Nobs{}.pickle'.format(Nobs)
    weight_path = 'saved_model/PINN_uh_weights_Telemac_Nobs{}.out'.format(Nobs)
    wmff_path = 'saved_model/PINN_uh_mff_Telemac_Nobs{}.out'.format(Nobs)

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
                ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path, wmffDir=wmff_path,
                useObs=useObs)

    endtimer = time.time()
    print ('PINN Time elapsed: ', endtimer-starttimer)

    # Test data
    Nt_test = Nt_train
    N_test = Nt_test * Nx    ## Nt_test x Nx
    X_test = X_star[:N_test,:]
    x_test = X_test[:,0:1]
    t_test = X_test[:,1:2]
    u_test = u_star[:N_test,:]
    h_test = h_star[:N_test,:]

    # Prediction
    u_pred, h_pred = model.predict(x_test, t_test)
    error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
    error_h = np.linalg.norm(h_test-h_pred,2)/np.linalg.norm(h_test,2)
    print('Error u: %e' % (error_u))
    print('Error h: %e' % (error_h))



    u_pred = u_pred.reshape([Nt_test, Nx])
    h_pred = h_pred.reshape([Nt_test, Nx])
    u_test = u_test.reshape([Nt_test, Nx])
    h_test = h_test.reshape([Nt_test, Nx])


    ## plotting
    hours = np.arange(len(time_model[:Nt_test]))
    xx, tt = np.meshgrid(x.flatten(), hours)

    
    plt.rcParams.update({'font.size': 18})
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    fig = plt.figure(figsize=(16.5, 16))
    gs = gridspec.GridSpec(2, 4, hspace=0.08, wspace=0.1)

    levels = np.linspace(0., 7, 9)
    ax0 = fig.add_subplot(gs[0, 1:3])
    cs = ax0.contourf(xx, tt, h_test[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax0.set_ylabel('Time (h)')
    ax0.set_xticklabels([])
    ax0.text(0.05, 0.9, '{}'.format('Reference (SWE)'), fontsize=16, transform=ax0.transAxes)

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(cs, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=14)
    cb.ax.yaxis.offsetText.set_fontsize(14)
    cb.set_label('Water depth (m)', fontsize=14)


   
    ax1 = fig.add_subplot(gs[1, :2])
    cs = ax1.contourf(xx, tt, h_pred[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    ax1.set_xlabel('Distance upstream (m)')
    ax1.set_ylabel('Time (h)')
    ax1.set_xticklabels([])
    ax1.text(0.05, 0.9, '{}'.format('PINN'), fontsize=16, transform=ax1.transAxes)

    levels_error = np.linspace(-0.25, 0.25, 11)
    ax2 = fig.add_subplot(gs[1, 2:])
    cs = ax2.contourf(xx, tt, h_test[:Nt_test,:]-h_pred[:Nt_test,:], cmap='bwr', levels=levels_error, alpha = 0.8, extend='both')
    ax2.set_xlabel('Distance upstream (m)')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(cs, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=14)
    cb.ax.yaxis.offsetText.set_fontsize(14)
    cb.set_label('Water depth error (m)', fontsize=14)

    plt.tight_layout()
    plt.savefig('figures/PINN_contour_comparison_Nobs{}.png'.format(Nobs))
    plt.close()


