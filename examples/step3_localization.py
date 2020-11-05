# -------------------------------------------------------------------
# An example of source localization after obtained the focal mechanism (ref: example_4_*).
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from examples.common_parameters import *
from examples.data_prepare import enquire_near_grid_points, load_sgt, load_data, load_loc_list_sta
from src.DSEM_Utils.DWidgets import get_proc_name
from src.DMTSolvers.DGridSearch_pnl_s import DFullgridSearch_pnl_s_MP

import pandas as pd
import numpy as np
import os




def error_by_location(eventid, initial_source, strike, dip, rake, mag, n_point):
    # observation waveform.
    waveform_data_path = str(waveform_database_dir) + str('evt_') + str(eventid) + str('.pkl')
    '''
    Please refer to Qinya Liu's Github program to prepare the waveform.
    https://github.com/liuqinya/py_cap  
    '''

    result_save_dir = str(proj_result_dir) + str("evt_") + str(eventid) + str('/') + str("localization/")
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir, exist_ok=True)

    output_file_path = str(result_save_dir) + str("localization.txt")
    log_file_path = str(result_save_dir) + str("log_location.txt")

    ''' Step:  load observed data and station location. '''
    disp3C_list, station_names, p_time_array = load_data(waveform_data_path)
    loc_list_sta = load_loc_list_sta(station_names)

    # p arrival tiem on data.
    n_p_pick_array = sampling_rate * p_time_array + n_event_offset
    # computing the approximate p and s travel time for cutting the SGT (synthetic waveform).
    n_tp_array = []
    n_ts_array = []

    ''' 
        Step: 
        (1) find a set of grid points close to the initial source location in the external denser grid file. 
        (2) Calculate the waveform misfit at the grid points by using the given focal mechanism (strike, dip, rake, magnitude) 
    '''

    # latitude, longitude, -> utm_x, utm_y
    # depth: km -> m
    zone_number = 11
    easting, northing, zone_number, zone_letter = from_latlon(latitude=initial_source[0],
                                                              longitude=initial_source[1],
                                                              force_zone_number=zone_number)

    source_utm = np.array([easting, northing, -1000.0 * initial_source[2]])

    #  find a set of grid points from the external pre-calculated denser grid file (eg: pre_calculated_grid.pkl).
    lat_arr, long_arr, z_arr, \
    utm_x_arr, utm_y_arr, utm_z_arr, \
    slice_index_arr, element_index_arr, \
    xi_arr, eta_arr, gamma_arr = enquire_near_grid_points(x=source_utm[0], y=source_utm[1], z=source_utm[2], n_points=n_point)

    res_lat = []
    res_long = []
    res_z = []
    res_strike = []
    res_dip=[]
    res_rake = []
    res_magnitude = []
    res_misfit = []
    res_segment = []

    max_searching_distance = 0
    for i in range(n_point):
        dist = 111.0 * np.sqrt(np.square(lat_arr[i] - initial_source[0]) + np.square(long_arr[i] - initial_source[1]))
        if dist > max_searching_distance:
            max_searching_distance = dist

    # loop to calculate the waveform misfit in space.
    for i in range(n_point):
        lat = lat_arr[i]
        long = long_arr[i]
        z = z_arr[i]
        utm_x = utm_x_arr[i]
        utm_y = utm_y_arr[i]
        utm_z = utm_z_arr[i]
        slice_index = slice_index_arr[i]
        element_index = element_index_arr[i]
        xi = xi_arr[i]
        eta = eta_arr[i]
        gamma = gamma_arr[i]


        loc_src = np.array([lat, long])
        proc_name = get_proc_name(slice_index)
        idx_element = element_index - 1  # MUST subtract 1. the index of element starts from 1 in the output_solver.txt.

        #
        ''' Step: Enquire the SGT at the grid point that is closest to the event location. '''
        list_sgt_array = load_sgt(station_names, proc_name, idx_element, xi, eta, gamma,
                                  sgt_database_dir, model_database_dir, NSPEC_global, nGLL_per_element)

        # calculate the arrival time of P and S for cutting the SGT.
        for loc_sta in loc_list_sta:
            dist = 111.0 * np.sqrt(np.square(loc_sta[0] - loc_src[0]) + np.square(loc_sta[1] - loc_src[1]))
            n_tp_array.append(int((dist / vp) * sampling_rate) + n_sgt_offset)
            n_ts_array.append(int((dist / vs) * sampling_rate) + n_sgt_offset)

        try:
            final_strike, final_dip, final_rake, final_magnitude, \
            minimum_misfit, n_segment, final_lags, final_ccs, \
            final_weights, final_wave_misift, final_syn_rtz_list, final_data_rtz_list \
                = DFullgridSearch_pnl_s_MP(
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
                strike_0=strike,
                strike_1=strike + 1,
                d_strike=10,
                dip_0=dip,
                dip_1=dip + 1,
                d_dip=10,
                rake_0=rake,
                rake_1=rake + 1,
                d_rake=10,
                mag_0=mag,
                mag_1=mag + 1,
                d_mag=10,
                PROCESSOR=PROCESSOR,
                log_saving_path=None, )

            res_lat.append(lat)
            res_long.append(long)
            res_z.append(z)
            res_strike.append(final_strike)
            res_dip.append(final_dip)
            res_rake.append(final_rake)
            res_magnitude.append(final_magnitude)
            res_misfit.append(minimum_misfit)
            res_segment.append(n_segment)

        except:
            print("Unable to plot!")

    # write out the searching result
    try:
        n_record = len(res_lat)
        with open(output_file_path, "w") as f:
            f.write("Latitude, Longitude, Z, strike, dip, rake, magnitude, misifit, n_segment\n")
            for i in range(n_record):
                text_str = str(res_lat[i]) + str(", ") \
                           + str(res_long[i]) + str(", ") \
                           + str(res_z[i]) + str(", ") \
                           + str(res_strike[i]) + str(", ") \
                           + str(res_dip[i]) + str(", ") \
                           + str(res_rake[i]) + str(', ') \
                           + str(res_magnitude[i]) + str(", ") \
                           + str(res_misfit[i]) + str(", ") \
                           + str(res_segment[i]) + str(" \n")
                f.write(text_str)
        print("Searching result saved at {}.".format(output_file_path))
    except:
        pass
        # print("Unable to write out the result at {}. ".format(output_file_path))


    # write out the log
    try:
        with open(log_file_path, 'w') as f:
            f.write(str("* Source localization\n"))
            f.write(str("* Working parameters\n"))
            f.write(str("EventID: ")+str(eventid)+str("\n"))
            f.write(str("Result path: ")+str(output_file_path) + str("\n"))
            f.write(str("Log path: ")+str(log_file_path)+str("\n"))

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
            f.write("Using the pre-defined focal mechanism: ")
            f.write(str("Initial location (lat, long z): ({}, {}, {}). \n".format(np.round(initial_source[0], 4), np.round(initial_source[1], 4), np.round(initial_source[2], 4))))
            f.write(str("Strike = {}. \n".format(strike)))
            f.write(str("Dip = {}. \n".format(dip)))
            f.write(str("Rake = {}. \n".format(rake)))
            f.write(str("Magnitude = {}. \n".format(mag)))
            f.write(str("Searching range = {} km.\n ".format(np.round(max_searching_distance, 4))))
    except:
        print("Unable to write parameters for event: {}.".format(eventid))



def find_location(eventid, b_segment=True):
    file_path = str(proj_result_dir) + str("evt_") + str(eventid) + str('/localization/') + str('localization.txt')
    save_file = str(proj_result_dir) + str("evt_") + str(eventid) + str('/localization/') + str('source.txt')
    # data format: Latitude, Longitude, Z, strike, dip, rake, magnitude, misifit, n_segment
    df = pd.read_csv(file_path)
    lat_arr = df.values[:, 0]
    long_arr = df.values[:, 1]
    z_arr = df.values[:, 2]
    strike_arr = df.values[:, 3]
    dip_arr = df.values[:, 4]
    rake_arr = df.values[:, 5]
    mag_arr = df.values[:, 6]
    misfit_arr = df.values[:, 7]
    data_segment = df.values[:, 8]
    if b_segment:
        # consider the waveform misift and the number of good segment.
        idx = np.argmax(1.0 / misfit_arr * data_segment)
    else:
        # ONLY consider the waveform misfit!
        idx = np.argmin(misfit_arr)

    with open(save_file, "w") as f:
        f.write("Latitude, Longitude, Z, strike, dip, rake, magnitude, misifit, n_segment\n")
        text_str = str(lat_arr[idx]) + str(", ") + str(long_arr[idx]) + str(", ") + str(z_arr[idx]) + str(", ")\
                   + str(strike_arr[idx]) + str(", ") + str(dip_arr[idx]) + str(", ") + str(rake_arr[idx]) + str(", ") \
                   + str( mag_arr[idx]) + str(", ") + str(misfit_arr[idx]) + str(", ") + str(data_segment[idx]) + str("\n")
        f.write(text_str)


    return lat_arr[idx], long_arr[idx], z_arr[idx], \
           strike_arr[idx], dip_arr[idx], rake_arr[idx],  mag_arr[idx],\
           misfit_arr[idx], data_segment[idx]








