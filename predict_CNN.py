import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tensorflow import keras
from keras.models import load_model
from keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

from myearth import findNearset1D
import othertime
from utils_ML import readslf

import pdb

def read_BC(filename):

    start_time = datetime(2011, 7, 30)  # Define the start time as specified

    data = pd.read_csv(filename, skiprows=3, delim_whitespace=True, names=['T', 'Q', 'SL'])

    data['T'] = data['T'].apply(lambda x: start_time + timedelta(seconds=x))

    times = pd.to_datetime(data['T'].values).to_pydatetime()
    discharges = data['Q'].to_numpy()
    water_levels = data['SL'].to_numpy()

    return times, discharges, water_levels



if __name__ == '__main__':


    Nes = [100, 200, 300, 400, 500, 600, 700, 800]

    starttime = datetime(2011, 8, 26)
    endtime = datetime(2011, 9, 5)

    plt.rcParams.update({'font.size': 18})
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    fig = plt.figure(figsize=(16.5, 14))
    gs = gridspec.GridSpec(5, 8, hspace=0.08, wspace=0.1, left=0.08, right=0.92, top=0.98, bottom=0.05)
    ax0 = fig.add_subplot(gs[0, 3:5])
    axes = [ax0]
    for i in range(4):
        ax_tmp1 = fig.add_subplot(gs[i+1, 0:2])
        ax_tmp2 = fig.add_subplot(gs[i+1, 2:4])
        ax_tmp3 = fig.add_subplot(gs[i+1, 4:6])
        ax_tmp4 = fig.add_subplot(gs[i+1, 6:8])
        axes += [ax_tmp1, ax_tmp2, ax_tmp3, ax_tmp4]

    for ii, Ne in enumerate(Nes):
        print ("Sample size is {}".format(Ne))

        BC_filename = 'files/mesh_1D_channel_hourly.liq'
        time_BC, Q_BC, wl_BC = read_BC(BC_filename)    

        ind0 = othertime.findNearest(starttime, time_BC)
        ind1 = othertime.findNearest(endtime, time_BC)

        time_BC = time_BC[ind0:ind1+1]
        Q_BC    = Q_BC[ind0:ind1+1]
        wl_BC   = wl_BC[ind0:ind1+1]


        Q_log = np.log1p(Q_BC)
        wl_log = np.log1p(wl_BC)
        scaler_Q = joblib.load('files/CNN/scaler_Q_Ne{}.pkl'.format(Ne))
        scaler_wl = joblib.load('files/CNN/scaler_wl_Ne{}.pkl'.format(Ne))

        Q_scaled = scaler_Q.transform(Q_log[None,:])
        wl_scaled = scaler_wl.transform(wl_log[None,:])
        features_new = np.stack((Q_scaled, wl_scaled), axis=-1)

        # Load the previously saved model
        model_cnn = load_model('files/CNN/cnn_Ne{}.keras'.format(Ne))

    
        predictions_new = model_cnn.predict(features_new)
        predictions_new_reshaped = predictions_new.reshape(-1, 241, 351)

        # Load h scaler
        scaler_h = joblib.load('files/CNN/scaler_h_Ne{}.pkl'.format(Ne))
        predictions_scaled = predictions_new_reshaped.reshape(-1)  # Flatten for inverse scaling
        h_inverse_scaled = scaler_h.inverse_transform(predictions_scaled[None,:]).flatten()
        h_inverse_scaled = h_inverse_scaled.reshape(-1, 241, 351)  # Reshape back

        # Inverse log transform
        h_final = np.expm1(h_inverse_scaled) 

        filename_mesh = 'files/mesh_1D_channel_dx100_update.slf'
        slf_filename = 'files/output_10days_hotstart.slf'
        x_values, elev, slope, time_model, water_surface, water_depth, u, v = readslf(slf_filename, filename_mesh, starttime, endtime)


        h_test = water_depth
        h_pred = h_final[0,:,:]
        error_h = np.linalg.norm(h_test-h_pred,2)/np.linalg.norm(h_test,2)
        print('Error h: %e' % (error_h))

        ## plotting
        hours = np.arange(len(time_model))
        xx, tt = np.meshgrid(x_values, hours)

        if ii == 0:
            levels = np.linspace(0., 7, 9)
            ax0 = axes[ii]
            cs = ax0.contourf(xx, tt, h_test[:,:], cmap='rainbow', levels=levels, alpha = 0.8)
            ax0.set_ylabel('Time (h)')
            ax0.set_xticklabels([])
            ax0.text(0.05, 0.9, '{} {}'.format(labels[ii], 'Reference'), fontsize=16, transform=ax0.transAxes)
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cb = fig.colorbar(cs, cax=cax, orientation='vertical')
            cb.ax.tick_params(labelsize=14)
            cb.ax.yaxis.offsetText.set_fontsize(14)
            cb.set_label('Water depth (m)', fontsize=14)

        ax1 = axes[ii*2+1]
        cs = ax1.contourf(xx, tt, h_pred[:,:], cmap='rainbow', levels=levels, alpha = 0.8)
        ax1.set_xticklabels([])
        ax1.text(0.05, 0.9, '{} N={}'.format(labels[ii+1], Ne), fontsize=16, transform=ax1.transAxes)
        ax1.text(0.05, 0.8, 'L2={:.2e}'.format(error_h), fontsize=16, transform=ax1.transAxes)

        levels_error = np.linspace(-0.25, 0.25, 11)
        ax2 = axes[ii*2+2]
        cs = ax2.contourf(xx, tt, h_test[:,:]-h_pred[:,:], cmap='bwr', levels=levels_error, alpha = 0.8, extend='both')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        if ii in [6,7]:
            ax1.set_xlabel('Distance upstream (m)')
            ax2.set_xlabel('Distance upstream (m)')

        if ii in [0,2,4,6]:
            ax1.set_ylabel('Time (h)')

        if ii in [1,3,5,7]:
            ax1.set_yticklabels([])
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cb = fig.colorbar(cs, cax=cax, orientation='vertical')
            cb.ax.tick_params(labelsize=14)
            cb.ax.yaxis.offsetText.set_fontsize(14)
            cb.set_label('Water depth error (m)', fontsize=14)

    plt.savefig('figures/CNN_contour_comparison.png')
    plt.close()
