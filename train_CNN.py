from netCDF4 import Dataset, num2date
import os
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import time
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense
import joblib

from myearth import findNearset1D
import othertime

import pdb


def calculate_metrics(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    mse = mean_squared_error(y_true, y_pred)  # Mean Squared Error
    mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_true, y_pred)  # R2 Score, coefficient of determination
    l2_error = np.sqrt(mse)  # L2 error is the square root of MSE
    return mse, mae, r2, l2_error


def evaluation(model, features_test, targets_test):

    loss_cnn = model.evaluate(features_test, targets_test.reshape(targets_test.shape[0], -1))

    # Predict with CNN
    predictions_cnn = model.predict(features_test)
    predictions_cnn = predictions_cnn.reshape(-1, Nt, Nx)  # Reshape predictions to match the original h shape

    ## Evaluation metrics
    targets_test_reshaped = targets_test.reshape(targets_test.shape[0], -1)
    predictions_cnn_reshaped = predictions_cnn.reshape(predictions_cnn.shape[0], -1)

    # Calculate metrics for CNN
    mse, mae, r2, l2 = calculate_metrics(targets_test_reshaped, predictions_cnn_reshaped)

    return mse, mae, r2, l2


if __name__ == '__main__':


    starttime = datetime(2011, 8, 26)
    endtime = datetime(2011, 9, 5)

    ## 10 year return period
    Q_threshold = 3500
    SS_threshold = 1.1

    filepath_ensemble = 'files/Telemac_output_ensemble_rp.nc'

    ## Number of samples used in the training
    Nes = [100, 200, 300, 400, 500, 600, 700, 800]

    time_training = []
    mse_all = []
    mae_all = []
    r2_all  = []
    l2_all  = []

    mse_min_all = []
    mae_min_all = []
    r2_min_all  = []
    l2_min_all  = []

    mse_max_all = []
    mae_max_all = []
    r2_max_all  = []
    l2_max_all  = []


    for Ne in Nes:
        print ("Sample size is {}".format(Ne))

        nc = Dataset(filepath_ensemble, 'r')
        print (nc)

        ens_ids = nc.variables['ens_ids'][:]
        timei_model = nc.variables['time']
        time_model = num2date(timei_model[:], timei_model.units, only_use_cftime_datetimes=False)
        x_values = nc.variables['x'][:]
        Q_BC = nc.variables['Discharge'][:]           # (Ne, Nt)
        wl_BC = nc.variables['WL'][:]                 # (Ne, Nt)
        water_depth = nc.variables['Water_Depth'][:]  # (Ne, Nt, Nx)
        Ne_tmp, Nt, Nx = water_depth.shape
        nc.close()

        Q_BC = Q_BC[:Ne,:]
        wl_BC = wl_BC[:Ne,:]
        water_depth = water_depth[:Ne,:,:]


        # scaling
        Q_log = np.log1p(Q_BC)
        wl_log = np.log1p(wl_BC)
        h_log = np.log1p(water_depth.reshape(Ne, -1))  # Reshape h to 2D for transformation

        scaler_Q = MinMaxScaler(feature_range=(0.01, 1))
        scaler_wl = MinMaxScaler(feature_range=(0.01, 1))
        scaler_h = MinMaxScaler(feature_range=(0.01, 1))


        Q_scaled = scaler_Q.fit_transform(Q_log)
        wl_scaled = scaler_wl.fit_transform(wl_log)
        h_scaled = scaler_h.fit_transform(h_log)
        joblib.dump(scaler_Q, 'files/CNN/scaler_Q_Ne{}.pkl'.format(Ne))
        joblib.dump(scaler_wl, 'files/CNN/scaler_wl_Ne{}.pkl'.format(Ne))
        joblib.dump(scaler_h, 'files/CNN/scaler_h_Ne{}.pkl'.format(Ne))

        h_scaled = h_scaled.reshape(Ne, Nt, Nx)  # Reshape back to original dimensions
        features = np.stack((Q_scaled, wl_scaled), axis=-1)  # Shape: (Ne, Nt, 2)

        # Splitting the dataset into training, validation, and test sets
        features_train, features_temp, targets_train, targets_temp = train_test_split(
            features, h_scaled, test_size=0.3, random_state=42
        )
        features_val, features_test, targets_val, targets_test = train_test_split(
            features_temp, targets_temp, test_size=0.5, random_state=42
        )

        ## Model Architecture 
        model_cnn = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(241, 2)),
            Conv1D(64, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(351 * 241)  # Adjust to the total number of outputs
        ])
        model_cnn.compile(optimizer='adam', loss='mse')

        start_time = time.time()

        # Training the CNN model
        model_cnn.fit(
            features_train, 
            targets_train.reshape(targets_train.shape[0], -1),  # Flatten targets for matching output shape
            epochs=10, 
            batch_size=10,
            validation_data=(features_val, targets_val.reshape(targets_val.shape[0], -1))
        )

        end_time = time.time()
        elapsed_time = end_time - start_time


        # Evaluate CNN
        mse_cnn, mae_cnn, r2_cnn, l2_cnn = evaluation(model_cnn, features_test, targets_test)

        model_cnn.save('files/CNN/cnn_Ne{}.keras'.format(Ne))

        print("CNN Model Metrics:")
        print(f"Mean Squared Error: {mse_cnn}")
        print(f"Mean Absolute Error: {mae_cnn}")
        print(f"R2 Score: {r2_cnn}")
        print(f"L2 Error: {l2_cnn}\n")

        time_training.append(elapsed_time)
        mse_all.append(mse_cnn)
        mae_all.append(mae_cnn)
        r2_all.append(r2_cnn)
        l2_all.append(l2_cnn)

        Q_test, wl_test = scaler_Q.inverse_transform(features_test[:,:,0]), scaler_wl.inverse_transform(features_test[:,:,1]) # (Ne, 241)
        Q_test, wl_test = np.expm1(Q_test), np.expm1(wl_test)
        Q_test, wl_test = np.max(Q_test,axis=1), np.max(wl_test, axis=1)
        SS_test = wl_test - 1.05 - 5 # 1.05 is maximum tide

        ind_min = np.where((Q_test<=Q_threshold)&(SS_test<=SS_threshold))[0]
        ind_max = np.where((Q_test>Q_threshold)&(SS_test>SS_threshold))[0]

        mse_min, mae_min, r2_min, l2_min = evaluation(model_cnn, features_test[ind_min,:,:], targets_test[ind_min,:,:])
        mse_max, mae_max, r2_max, l2_max = evaluation(model_cnn, features_test[ind_max,:,:], targets_test[ind_max,:,:])

        mse_min_all.append(mse_min)
        mae_min_all.append(mae_min)
        r2_min_all.append(r2_min)
        l2_min_all.append(l2_min)

        mse_max_all.append(mse_max)
        mae_max_all.append(mae_max)
        r2_max_all.append(r2_max)
        l2_max_all.append(l2_max)


        del model_cnn
        keras.backend.clear_session()

    data = zip(time_training, mse_all, mae_all, r2_all, l2_all, mse_min_all, mae_min_all, r2_min_all, l2_min_all, mse_max_all, mae_max_all, r2_max_all, l2_max_all)

    # Write to a CSV file
    with open('files/metrics_CNN.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time Training', 'MSE', 'MAE', 'R2', 'L2', 'MSE_min', 'MAE_min', 'R2_min', 'L2_min', 'MSE_max', 'MAE_max', 'R2_max', 'L2_max'])  # Writing headers
        writer.writerows(data)
