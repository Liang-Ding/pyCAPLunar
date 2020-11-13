# -------------------------------------------------------------------
# Focal mechanism inversion using grid search method.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from examples.common_parameters import *
from examples.data_prepare import enquire_near_grid_points, load_sgt, load_data, load_loc_list_sta

from src.DSEM_Utils.DWidgets import get_proc_name
from src.DMTSolvers.DFMSolver import DFMGridSearch

import numpy as np
from datetime import datetime
import time



def example_fm_inversion():
    eventid = 11058623
    initial_source = np.array([36.6758, -1174853, 8530])
    strike_0 = 180.0
    strike_1 = 360.0
    d_strike = 10.0
    dip_0 = 0.0
    dip_1 = 90.0
    d_dip = 10.0
    rake_0 = 0
    rake_1 = 360
    d_rake = 10.0
    mag_0 = 4.4  # initial magnitude.
    mag_1 = 4.5
    d_mag = 10

    # first grid search
    print("! Doing grid-search ... ")
    gridsearch_fm(eventid, initial_source,
                  strike_0, strike_1, d_strike,
                  dip_0, dip_1, d_dip,
                  rake_0, rake_1, d_rake,
                  mag_0, mag_1, d_mag)


def gridsearch_fm(eventid, initial_source,
                  strike_0, strike_1, d_strike,
                  dip_0, dip_1, d_dip,
                  rake_0, rake_1, d_rake,
                  mag_0, mag_1, d_mag):

    # observation waveform.
    waveform_data_path = str(waveform_database_dir) + str('evt_') + str(eventid) + str('.pkl')

    # data output path.
    project_dir = str(proj_result_dir) + str("evt_") + str(eventid) + str('/')


    ''' Step:  load observed data and station location. '''
    data_array_list, station_names, p_time_array = load_data(waveform_data_path)
    loc_sta_list = load_loc_list_sta(station_names)


    lat, long, z, \
    utm_x, utm_y, utm_z, \
    slice_index, element_index, \
    xi, eta, gamma = enquire_near_grid_points(x=initial_source[0], y=initial_source[1], z=initial_source[2], n_points=1)

    loc_src = np.array([lat, long])
    # the name of processor
    proc_name = get_proc_name(slice_index)
    idx_element = element_index - 1  # MUST subtract 1. the index of element starts from 1 in the output_solver.txt.

    #
    ''' Step: Enquire the SGT at the grid point that is closest to the event location. '''
    sgt_array_list = load_sgt(station_names, proc_name, idx_element, xi, eta, gamma,
                              sgt_database_dir, model_database_dir, NSPEC_global, nGLL_per_element)

    # p arrival tiem on data.
    n_picked_tp_array = sampling_rate * p_time_array + n_event_offset

    # computing the approximate p and s travel time for cutting the SGT (synthetic waveform).
    n_computed_tp_array = []
    n_computed_ts_array = []

    for loc_sta in loc_sta_list:
        dist = 111.0 * np.sqrt(np.square(loc_sta[0] - loc_src[0]) + np.square(loc_sta[1] - loc_src[1]))
        n_computed_tp_array.append(int((dist / vp) * sampling_rate) + n_sgt_offset)
        n_computed_ts_array.append(int((dist / vs) * sampling_rate) + n_sgt_offset)


    t0 = time.time()
    job_identifier = "search_fm_1st"
    print("! Inversion Start at {} ... ".format(datetime.now()))
    try:
        # first grid search
        dfs = DFMGridSearch(sgt_array_list, n_computed_tp_array, n_computed_ts_array,
                 data_array_list, n_picked_tp_array, loc_src, loc_sta_list,
                 n_pnl_length, n_pnl_back_offset, n_surface_length, n_surface_back_offset,
                 b_pnl_filter, pnl_freq_min, pnl_freq_max,
                 b_surface_filter, surface_freq_min, surface_freq_max, df,
                 misfit_threshold, timeshift_threshold, cc_threshold, n_valid_waveform_threshold,
                 strike_0, strike_1, d_strike,
                 dip_0, dip_1, d_dip,
                 rake_0, rake_1, d_rake,
                 mag_0, mag_1, d_mag,
                 project_dir, job_identifier,
                 b_write_each_step=True,
                 b_write_working_log=True,
                 b_write_best_result_file=True)


        print("*" * 10)
        print("Search focal mechanism ... ")
        strike, dip, rake, magnitude, misfit, n_valid_waveform = dfs.get_focal_mechanism()
        print("! strike = ", strike)
        print("! dip = ", dip)
        print("! rake = ", rake)
        print("! magnitude = ", magnitude)
        print("! misfit = ", misfit)
        print("! valid waveform = ", n_valid_waveform)
        print("! Time elapse = {} s.\n".format(time.time() - t0))


        print('\n')
        print("*"*10)
        print("Search Magnitude ...")
        # search magnitude
        strike_range = np.array([strike])
        dip_range = np.array([dip])
        rake_range = np.array([rake])
        mag_range = np.arange(mag_left, mag_right, mag_step)


        dfs.set_strike_range(strike_range)
        dfs.set_dip_range(dip_range)
        dfs.set_rake_range(rake_range)
        dfs.set_magnitude_range(mag_range)
        dfs.set_job_identifier(job_identifier=str('search_magnitude'))

        strike, dip, rake, magnitude, misfit, n_valid_waveform = dfs.get_focal_mechanism()
        print("! strike = ", strike)
        print("! dip = ", dip)
        print("! rake = ", rake)
        print("! magnitude = ", magnitude)
        print("! misfit = ", misfit)
        print("! valid waveform = ", n_valid_waveform)
        print("! Time elapse = {} s.\n".format(time.time() - t0))

        print('\n')
        print("*"*10)
        print("Search best focal mechanism...")
        # second grid search
        strike_range = np.hstack([np.arange(strike - strike_left, strike, strike_step),
                                  np.arange(strike + 1, strike + strike_right, strike_step)])
        dip_range = np.hstack([np.arange(dip - dip_left, dip, dip_step),
                                  np.arange(dip + 1, dip + dip_right, dip_step)])
        rake_range = np.hstack([np.arange(rake - rake_left, rake, rake_step),
                               np.arange(rake + 1, rake + rake_right, rake_step)])
        mag_range = np.array([magnitude])

        dfs.set_strike_range(strike_range)
        dfs.set_dip_range(dip_range)
        dfs.set_rake_range(rake_range)
        dfs.set_magnitude_range(mag_range)
        dfs.set_job_identifier(job_identifier=str('search_fm_2nd'))
        strike, dip, rake, magnitude, misfit, n_valid_waveform = dfs.get_focal_mechanism()
        print("! strike = ", strike)
        print("! dip = ", dip)
        print("! rake = ", rake)
        print("! magnitude = ", magnitude)
        print("! misfit = ", misfit)
        print("! valid waveform = ", n_valid_waveform)
        print("! Time elapse = {} s.\n".format(time.time() - t0))

    except:
        print("INVERSION ABORT UNEXPECTEDLY !")

    print("Time elapse = {} s. ".format(time.time() - t0))



if __name__ == '__main__':
    example_fm_inversion()

