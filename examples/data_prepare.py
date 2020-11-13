# -------------------------------------------------------------------
# Functions to prepare the data for the inversion.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from src.DMath.interp_tools import DCreate_anchors_xi_eta_gamma, DLagrange_interp_sgt, DLagrange_any3D
from src.DSEM_Utils.ibool_reader import DEnquire_Element
from src.DSEM_Utils.sgt_reader import DEnquire_SGT

import pandas as pd
import numpy as np
import pickle


def load_data(file_path):
    '''Return the seismograms, station name, p_picktime.'''
    with open(file_path, 'rb') as f:
        disp3C_list = pickle.load(f)
        station_names = pickle.load(f)
        p_time_array = pickle.load(f)
    return disp3C_list, station_names, p_time_array


def load_sgt(station_names, proc_name, idx_element, xi, eta, gamma,
             sgt_database_dir, model_database_dir, NSPEC_global, nGLL_per_element):
    '''
    * Enquire the SGT data at the selected grid point.

    :param station_names:       The name of the stations that the SGT database has been computed.
    :param proc_name:           The name of the process where the grid point locates.
    :param idx_element:         The index of the element where the grid point locates.
    :param xi:                  The xi of the grid point.
    :param eta:                 The eta of the grid point.
    :param gamma:               The gamma of the point.
    :param sgt_database_dir:    The directory of the sgt database.
    :param model_database_dir:  The directory of the model.
    :param NSPEC_global:        The number of element in one slice (processor).
    :param nGLL_per_element:    The number of GLL in each element saved in the SGT database.
    :return:
            The list of the SGT data extracted and sorted according to the station_names.
    '''

    list_sgt_array = []
    for station in station_names:
        ibool_file = str(model_database_dir) + proc_name + "_ibool.bin"
        sgt_data_path = str(sgt_database_dir) + str(station) + str('/') + str(proc_name) + str("_sgt_data.bin")
        sgt_info_path = str(sgt_database_dir) + str(station) + str('/') + str(proc_name) + str("_sgt_info.pkl")

        # enquire the global index of GLL points
        gll_points = DEnquire_Element(ibool_file, NSPEC_global, idx_element, nGLL_per_element=nGLL_per_element)
        # enquire SGT at the stored GLL points.
        sgt_arr_list = DEnquire_SGT(data_path=sgt_data_path, info_path=sgt_info_path, GLL_points=gll_points, encoding_level=8)

        # prepare interpolation
        xi_gll, eta_gll, gamma_gll = DCreate_anchors_xi_eta_gamma(ngll_xyz=3)
        h_xi_arr, h_eta_arr, h_gamma_arr = DLagrange_any3D(xi, eta, gamma, xi_gll, eta_gll, gamma_gll)
        if 27 == nGLL_per_element:
            ngll_x = 3
            ngll_y = 3
            ngll_z = 3
        else:
            ngll_x = 5
            ngll_y = 5
            ngll_z = 5

        sgt_interp = DLagrange_interp_sgt(h_xi_arr, h_eta_arr, h_gamma_arr, sgt_arr_list,
                                          ngll_x=ngll_x, ngll_y=ngll_y, ngll_z=ngll_z)
        sgt_interp = sgt_interp[:-1, :, :]
        list_sgt_array.append(sgt_interp)
    return list_sgt_array



def load_loc_list_sta(station_names):
    '''Load the location of selected stations.'''
    station_file = str('STATIONS.txt')
    df = pd.read_csv(station_file, sep=' ')
    all_station = list(df.values[:, 0])
    lat_list = df.values[:, 2]
    long_list = df.values[:, 3]

    loc_list_sta = []
    for sta in station_names:
        idx = all_station.index(sta)
        tmp = []
        tmp.append(lat_list[idx])
        tmp.append(long_list[idx])
        loc_list_sta.append(tmp)
    return loc_list_sta




''' Grid operation '''
def grid_information():
    '''
    * Read the pre-computed denser grid file.
    '''

    file_path = "pre_calculated_grid.pkl"
    with open(file_path, 'rb') as f:
        station_names = pickle.load(f)
        lat_array = pickle.load(f)
        long_array = pickle.load(f)
        z_array = pickle.load(f)
        utm_x_array = pickle.load(f)
        utm_y_array = pickle.load(f)
        utm_z_array = pickle.load(f)
        slice_index_array = pickle.load(f)
        element_index_array = pickle.load(f)
        xi_array = pickle.load(f)
        eta_array = pickle.load(f)
        gamma_array = pickle.load(f)

    return station_names, lat_array, long_array, z_array, \
    utm_x_array, utm_y_array, utm_z_array, \
    slice_index_array, element_index_array, \
    xi_array, eta_array, gamma_array



def enquire_near_grid_points(x, y, z, n_points, method='LATLONGZ'):
    '''
    * Enquire the information of the grid point that closest to the given point.

    :param x:       The latitude.
    :param y:       The longitude.
    :param z:       The elevation.
    :param method:  The type of coordination. Either 'LATLONGZ' or 'UTM'.
    :return:
            The information of the grid point that closest to the given point.
    '''

    station_names, lat_array, long_array, z_array, \
    utm_x_array, utm_y_array, utm_z_array, \
    slice_index_array, element_index_array, \
    xi_array, eta_array, gamma_array = grid_information()

    n_points = int(n_points)
    if str('LATLONGZ') == str(method).upper():
        distance_arr = np.sqrt(np.power(lat_array - x, 2) + np.power(long_array - y, 2) + np.power(z_array - z, 2))
    elif str('UTM') == str(method).upper():
        distance_arr = np.sqrt(np.power(utm_x_array - x, 2) + np.power(utm_y_array - y, 2) + np.power(utm_z_array - z, 2))

    if 1 == n_points:
        # return values
        idx = np.argmin(distance_arr)
    else:
        # return array.
        idx = np.argpartition(distance_arr, n_points)[:n_points]
        # print("idx = ", idx)

    return lat_array[idx], long_array[idx], z_array[idx], \
           utm_x_array[idx], utm_y_array[idx], utm_z_array[idx], \
           slice_index_array[idx], element_index_array[idx], \
           xi_array[idx], eta_array[idx], gamma_array[idx]




def enquire_grid_points_cubic(x_min, x_max, y_min, y_max, z_min, z_max, model='LATLONGZ'):
    station_names, lat_array, long_array, z_array, \
    utm_x_array, utm_y_array, utm_z_array, \
    slice_index_array, element_index_array, \
    xi_array, eta_array, gamma_array = grid_information()

    idx = []
    n_data = len(lat_array)
    if str(model).upper() == str('LATLONGZ'):
        for i in range(n_data):
            if (lat_array[i] >= x_min) & (lat_array[i] <= x_max) & \
                    (long_array[i] >= y_min) & (long_array[i] <= y_max) & \
                    (z_array[i] >= z_min) & (z_array[i] <= z_max):
                idx.append(i)

    elif str(model).upper() == str('UTM'):
        for i in range(n_data):
            if (utm_x_array[i] >= x_min) & (utm_x_array[i] <= x_max) & \
                    (utm_y_array[i] >= y_min) & (utm_y_array[i] <= y_max) & \
                    (utm_z_array[i] >= z_min) & (utm_z_array[i] <= z_max):
                idx.append(i)

    return lat_array[idx], long_array[idx], z_array[idx], \
           utm_x_array[idx], utm_y_array[idx], utm_z_array[idx], \
           slice_index_array[idx], element_index_array[idx], \
           xi_array[idx], eta_array[idx], gamma_array[idx]













