# -------------------------------------------------------------------
# Calculate the waveform misfit by using the grid search.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from examples.common_parameters import *
from examples.data_prepare import enquire_near_grid_points, load_sgt, load_data, load_loc_list_sta
from src.DSEM_Utils.DWidgets import get_proc_name
from src.DMTSolvers.DGridSearch_pnl_s import DFullgridSearch_pnl_s_MP

import numpy as np
import os


def gridsearch_fm(eventid, initial_source,
                  strike_0, strike_1, d_strike,
                  dip_0, dip_1, d_dip,
                  rake_0, rake_1, d_rake,
                  mag_0, mag_1, d_mag, job_index="gridsearch"):

    # observation waveform.
    waveform_data_path = str(waveform_database_dir) + str('evt_') + str(eventid) + str('.pkl')
    '''
    Please refer to Qinya Liu's Github program to prepare the waveform.
    https://github.com/liuqinya/py_cap  
    '''

    # set the output folder.
    event_result_dir = str(proj_result_dir) + str("evt_") + str(eventid) + str('/')
    saving_dir = str(event_result_dir) + str('gsfull_fm_pnl_s/')
    output_dir = str(proj_result_dir) + str("output") + str('/')


    '''check saving directory - make directory'''
    # saving directory of the project.
    if not os.path.exists(proj_result_dir):
        os.makedirs(proj_result_dir, exist_ok=True)
    # the saving directory of the event.
    if not os.path.exists(event_result_dir):
        os.makedirs(event_result_dir, exist_ok=True)
    # the saving directory of the searching results.
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir, exist_ok=True)
    # output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    ''' Step:  load the observed data, the station location, and the first arrival time of P at each station. '''
    disp3C_list, station_names, p_time_array = load_data(waveform_data_path)
    loc_list_sta = load_loc_list_sta(station_names)
    # the first arrival of the p wave in sample.
    n_p_pick_array = sampling_rate * p_time_array + n_event_offset


    ''' 
    Step: 
    enquire the SGT data at the grid point that is close to the initial source. 
    '''
    # Find the grid point that close to the initial source
    # from the external pre-calculated denser grid file (eg: pre_calculated_grid.pkl).
    lat, long, z, \
    utm_x, utm_y, utm_z, \
    slice_index, element_index, \
    xi, eta, gamma = enquire_near_grid_points(x=initial_source[0], y=initial_source[1], z=initial_source[2], n_points=1)

    loc_src = np.array([lat, long])
    proc_name = get_proc_name(slice_index)
    idx_element = element_index - 1  # MUST subtract 1. the index of element starts from 1 in the output_solver.txt.

    # (3). enquire sgt at the grid point.
    list_sgt_array = load_sgt(station_names, proc_name, idx_element, xi, eta, gamma,
                              sgt_database_dir, model_database_dir, NSPEC_global, nGLL_per_element)


    # computing the approximate p and s travel time for cutting the SGT (synthetic waveform).
    n_tp_array = []
    n_ts_array = []
    for loc_sta in loc_list_sta:
        dist = 111.0 * np.sqrt(np.square(loc_sta[0] - loc_src[0]) + np.square(loc_sta[1] - loc_src[1]))
        n_tp_array.append(int((dist / vp) * sampling_rate) + n_sgt_offset)
        n_ts_array.append(int((dist / vs) * sampling_rate) + n_sgt_offset)

    log_saving_path = str(saving_dir) + str('job_')+str(job_index)+str(".txt")
    try:
        DFullgridSearch_pnl_s_MP(
            station_names,
            list_sgt_array, n_tp_array, n_ts_array,
            disp3C_list, n_p_pick_array,
            loc_src, loc_list_sta,
            p_waveform_length, p_n_offset,
            s_waveform_length, s_n_offset,
            b_p_filter, p_freqmin, p_freqmax,
            b_s_filter, s_freqmin, s_freqmax, df,
            n_component=n_component,
            minimum_waveform_segment=minimum_waveform_segment,
            lag_threshold=lag_threshold,
            cc_threshold=cc_threshold,
            misfit_threshold=misfit_threshold,
            strike_0=strike_0,
            strike_1=strike_1,
            d_strike=d_strike,
            dip_0=dip_0,
            dip_1=dip_1,
            d_dip=d_dip,
            rake_0=rake_0,
            rake_1=rake_1,
            d_rake=d_rake,
            mag_0=mag_0,
            mag_1=mag_1,
            d_mag=d_mag,
            PROCESSOR=PROCESSOR,
            log_saving_path=log_saving_path)

    except:
        print("unable to search!")

    # Write out the parameters
    try:
        log_file = str(saving_dir) + str('job_')+str(job_index)+ str("_parameters.txt")
        with open(log_file, 'w') as f:
            f.write(str("* Working parameters\n"))
            f.write(str("EventID: ")+str(eventid)+str("\n"))
            f.write(str("Job idetifier: ")+str(job_index) + str("\n"))
            f.write(str("Saving directory: ")+str(saving_dir)+str("\n"))

            f.write("\n")
            f.write(str("* Model and database\n"))
            f.write(str("The Solver using Strain Green's tensor.\n"))
            f.write(str("Number of element per slice = ")+str(NSPEC_global) + str("\n"))
            f.write(str("Number of GLL points saved each element = ")+str(nGLL_per_element)+str("\n"))
            f.write(str("Sampling interval = ")+str(np.around(1.0/sampling_rate, 4))+str(" s. \n"))
            f.write(str("Offset of the SGT database (The time gap between the data begin and the SGT begin) =") + str(
                sgt_offset) + str(" s. \n"))
            f.write(str("Velocity to estimate the arrival time of P in SGT = {} m/s. \n".format(vp)))
            f.write(str("Velocity to estimate the arrival time of S in SGT = {} m/s. \n".format(vs)))

            f.write("\n")
            f.write(str("* Waveforms\n"))
            f.write(str("Pnl length = ") + str(np.round(p_waveform_length/sampling_rate, 4))+str(" s. \n"))
            f.write(str("Pnl offset (The time gap between the data begin and phase begin) = ") + str(p_n_offset) + str(
                " s. \n"))
            f.write(str("S/Surface length = ") + str(np.round(s_waveform_length/sampling_rate, 4))+str(" s. \n"))
            f.write(str("Pnl offset (The time gap between the data begin and phase begin) = ") + str(s_n_offset) + str(
                " s. \n"))

            f.write("\n")
            f.write(str("* FILTERS\n"))
            f.write("Filter type: Band-pass\n")
            f.write("Frequency range for pnl: {} - {}s (Equivlent to {} - {} Hz)\n".format(p_Tmin, p_Tmax, np.round(p_freqmin, 4),
                                                                       np.round(p_freqmax, 4)))

            f.write("\n")
            f.write(str("* THRESHOLD\n"))
            f.write(str("Threshold of CC = ")+str(cc_threshold)+str(". \n"))
            f.write(str("Threshold of time shift = ") + str(lag_threshold) + str(" s. \n"))
            f.write(str("Threshold of misift = ") + str(misfit_threshold) + str("s. \n"))
            f.write(str("The minimum number of acceptable waveform = ") + str(minimum_waveform_segment) + str("\n"))

            f.write("\n")
            f.write("* Solver\n")
            f.write("Searching method: Grid search!\n")
            f.write(str("Strike: [{}, {}], step: {}. \n".format(strike_0, strike_1, d_strike)))
            f.write(str("Dip: [{}, {}], step: {}. \n".format(dip_0, dip_1, d_dip)))
            f.write(str("Rake: [{}, {}], step: {}. \n".format(rake_0, rake_1, d_rake)))
            f.write(str("Magnitude: [{}, {}], step: {}. \n".format(mag_0, mag_1, d_mag)))

    except:
        print("Unable to write parameters for event: {}.".format(eventid))








