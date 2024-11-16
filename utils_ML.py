from datetime import datetime
import numpy as np
import h5py

import sys
sys.path.append('/qfs/people/feng779/PYTHON/utils')
import othertime

sys.path.append('/qfs/people/feng779/TELEMAC/telemac-mascaret/scripts/python3')
from data_manip.extraction.telemac_file import TelemacFile
from data_manip.formats.regular_grid import interpolate_on_grid
from data_manip.computation.datetimes import compute_datetimes

import pdb

def time_convert(intime):
    """
    function to convert the time from string to datetime
    """
    Nt = intime.shape[0]
    outtime = []
    for t in range(Nt):
        timestr = intime[t].decode('utf-8')
        outtime.append(datetime.strptime(timestr, '%d%b%Y %H:%M:%S'))

    return outtime


def readhdf(filepath, starttime, endtime):

    hf = h5py.File(filepath,'r')

    attrs = hf['Geometry']['2D Flow Areas']['Attributes'][:]
    print (attrs)

    # geometry_group = hf['Geometry/2D Flow Areas']
    # xyv = geometry_group['Cell Points'][:]
    outline_group = hf['Geometry/2D Flow Areas/outline']
    available_keys = list(outline_group.keys())
    print("Available keys in 'outline' group:", available_keys)

    manning_n = outline_group["Cells Center Manning's n"][:]
    elev = outline_group['Cells Minimum Elevation'][:]
    xyc = outline_group["Cells Center Coordinate"][:] ## coordinate of cell center and domain boundary


    group_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/outline'
    group = hf[group_path]

    # List all members in the group
    members = list(group.keys())

    # Print the members
    print("Members under the specified group:")
    for member in members:
        print(member)

    Timestamp = hf['Results']['Unsteady']['Output']["/Results/Unsteady/Output"]['Output Blocks']['Base Output']["Unsteady Time Series"]['Time Date Stamp'][:]
    time_model = time_convert(Timestamp)
    water_surface = group['Water Surface'][:]
    u = group['Cell Velocity - Velocity X'][:]
    v = group['Cell Velocity - Velocity Y'][:]
    hf.close()

    ind0 = othertime.findNearest(starttime, time_model)
    ind1 = othertime.findNearest(endtime, time_model)
    time_model = time_model[ind0:ind1+1]
    water_surface = water_surface[ind0:ind1+1, :]
    u = u[ind0:ind1+1, :]     
    v = v[ind0:ind1+1, :]


    ## 2D domain above, only take values at the 2D channel centerline x=1000 m
    x_target = 1000
    tolerance = 101  # Define a tolerance for x selection
    # Filter data to get close values around x_target
    mask = (xyc[:, 0] > x_target - tolerance) & (xyc[:, 0] < x_target + tolerance)
    x_values = xyc[mask, 1] ## along-channel coordinate, downstream: x=0, upstream: x=70000
    water_surface = water_surface[:,mask]
    u = u[:,mask]
    v = v[:,mask]
    elev = elev[mask]

    water_depth = water_surface - elev


    indices = np.argsort(x_values)
    x_values = x_values[indices]
    elev = elev[indices]
    water_surface = water_surface[:, indices]
    water_depth = water_depth[:, indices]
    u = u[:, indices]
    v = v[:, indices]

    ## remove values at the boundary
    x_values = x_values[1:-1]
    elev = elev[1:-1]
    water_surface = water_surface[:,1:-1]
    water_depth = water_depth[:,1:-1]
    u = u[:,1:-1]
    v = v[:,1:-1]

    slope = np.gradient( elev, x_values)
    
    return x_values, elev, slope, time_model, water_surface, water_depth, u, v


def readslf(filepath, filename_mesh, starttime, endtime):
    """
    read Telemac output
    """

    res = TelemacFile(filename_mesh)
    elev = res.get_data_value('BOTTOM', 0)
    res.close()

    res = TelemacFile(filepath)
    print("Output variables", res.get_data_var_list())
    time_model = compute_datetimes(res.times, initial_date=res.datetime)
    Nt = len(time_model)
    x = res.tri.x
    y = res.tri.y
    Nv = len(x)

    u = np.zeros([Nt, Nv])
    v = np.zeros([Nt, Nv])
    water_depth = np.zeros([Nt, Nv])
    water_surface = np.zeros([Nt, Nv])

    for t in range(Nt):
        u[t,:] = res.get_data_value('VELOCITY U', t)
        v[t,:] = res.get_data_value('VELOCITY V', t)
        water_depth[t,:] = res.get_data_value('WATER DEPTH', t)
        water_surface[t,:] = res.get_data_value('FREE SURFACE', t)

    ind0 = othertime.findNearest(starttime, time_model)
    ind1 = othertime.findNearest(endtime, time_model)

    time_model = time_model[ind0:ind1+1]
    water_depth = water_depth[ind0:ind1+1, :]
    water_surface = water_surface[ind0:ind1+1, :]
    u = u[ind0:ind1+1, :]
    v = v[ind0:ind1+1, :]

    ## 2D domain above, only take values at the 2D channel centerline x=1000 m
    x_target = 1000
    tolerance = 10  # Define a tolerance for x selection
    # Filter data to get close values around x_target
    mask = (x > x_target - tolerance) & (x < x_target + tolerance)
    x_values = y[mask] ## along-channel coordinate, downstream: x=0, upstream: x=70000
    water_depth = water_depth[:,mask]
    water_surface = water_surface[:,mask]
    u = u[:,mask]
    v = v[:,mask]
    elev = elev[mask]


    indices = np.argsort(x_values)
    x_values = x_values[indices]
    elev = elev[indices]
    water_depth = water_depth[:, indices]
    water_surface = water_surface[:, indices]
    u = u[:, indices]
    v = v[:, indices]

    slope = np.gradient( elev, x_values)

    return x_values, elev, slope, time_model, water_surface, water_depth, u, v
