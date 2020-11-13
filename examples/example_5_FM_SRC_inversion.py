# -------------------------------------------------------------------
# Simultaneously determine the focal mechanism and source location.
# based on grid search method.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from examples.common_parameters import *
from examples.data_prepare import enquire_near_grid_points, load_sgt, load_data, load_loc_list_sta, enquire_grid_points_cubic


from src.DSEM_Utils.DWidgets import get_proc_name
from src.DMTSolvers.DFMSolver import DFMGridSearch

import numpy as np
from datetime import datetime
import time


def example_source_inversion():
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
    source_inversion(eventid, initial_source,
                  strike_0, strike_1, d_strike,
                  dip_0, dip_1, d_dip,
                  rake_0, rake_1, d_rake,
                  mag_0, mag_1, d_mag)



def source_localization(dfs, initial_source, station_names, loc_list_sta):
    '''Source localization'''
    print('\n')
    print("*" * 10)
    print("Search source location...")
    # enquire a set of point from the pre-computed mesh.

    output_file_path = dfs.save_step_dir + str("source_localization.txt")
    log_file_path = dfs.save_step_dir + str("source_localization_log.txt")
    result_file_path = dfs.project_dir + str("inversion_result.txt")

    dfs.set_write_best_result(False)
    dfs.set_write_searching_step(False)
    dfs.set_write_best_result(False)

    lat_arr, long_arr, z_arr, \
    utm_x_arr, utm_y_arr, utm_z_arr, \
    slice_index_arr, element_index_arr, \
    xi_arr, eta_arr, gamma_arr = enquire_grid_points_cubic(x_min=initial_source[0] - x_min_range,
                                                           x_max=initial_source[0] + x_max_range,
                                                           y_min=initial_source[1] - y_min_range,
                                                           y_max=initial_source[1] + y_max_range,
                                                           z_min=initial_source[2] * -1000 - z_min_range,
                                                           z_max=initial_source[2] * -1000 + z_max_range,
                                                           model=location_range_model)

    n_point = len(lat_arr)
    tmp_lat = []
    tmp_long = []
    tmp_z = []
    tmp_strike = []
    tmp_dip = []
    tmp_rake = []
    tmp_magnitude = []
    tmp_misfit = []
    tmp_segment = []

    result_lat = 0
    result_long = 0
    result_z = 0
    result_strike = 0
    result_dip = 0
    result_rake = 0
    result_magnitude = 0
    result_misfit = 0
    result_segment = 0

    max_searching_distance = 0
    for i in range(n_point):
        dist = 111.0 * np.sqrt(np.square(lat_arr[i] - initial_source[0]) + np.square(long_arr[i] - initial_source[1]))
        if dist > max_searching_distance:
            max_searching_distance = dist

    # get the previously obtained result.
    strike, dip, rake, magnitude, misfit, n_valid_waveform = dfs.get_focal_mechanism()

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

        n_tp_array = []
        n_ts_array = []

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

        # set sgt, set computed tp and ts.
        dfs.set_Strain_Greens_Function(list_sgt_array)
        dfs.set_computed_tp_array(n_tp_array)
        dfs.set_computed_ts_array(n_ts_array)
        dfs.set_source_location(loc_src)

        try:
            # search magnitude
            strike_range = np.array([strike])
            dip_range = np.array([dip])
            rake_range = np.array([rake])
            mag_range = np.array([magnitude])

            dfs.set_strike_range(strike_range)
            dfs.set_dip_range(dip_range)
            dfs.set_rake_range(rake_range)
            dfs.set_magnitude_range(mag_range)
            dfs.set_job_identifier(job_identifier=str('localization'))

            dfs.reset_current_misfit()
            _, _, _, _, misfit, n_valid_waveform = dfs.get_focal_mechanism()

            tmp_lat.append(lat)
            tmp_long.append(long)
            tmp_z.append(z)
            tmp_strike.append(strike)
            tmp_dip.append(dip)
            tmp_rake.append(rake)
            tmp_magnitude.append(magnitude)
            tmp_misfit.append(misfit)
            tmp_segment.append(n_valid_waveform)

        except:
            print("Unable to plot!")

    dfs.reset_current_misfit()
    try:
        n_record = len(tmp_lat)
        with open(output_file_path, "w") as f:
            f.write("Latitude, Longitude, Z, strike, dip, rake, magnitude, misifit, n_segment\n")
            for i in range(n_record):
                if dfs.evaluate_solution(tmp_misfit[i], tmp_segment[i]):
                    result_lat = tmp_lat[i]
                    result_long = tmp_long[i]
                    result_z = tmp_z[i]
                    result_strike = tmp_strike[i]
                    result_dip = tmp_dip[i]
                    result_rake = tmp_rake[i]
                    result_magnitude = tmp_magnitude[i]
                    result_misfit = tmp_misfit[i]
                    result_segment = tmp_segment[i]

                text_str = str(tmp_lat[i]) + str(", ") \
                           + str(tmp_long[i]) + str(", ") \
                           + str(tmp_z[i]) + str(", ") \
                           + str(tmp_strike[i]) + str(", ") \
                           + str(tmp_dip[i]) + str(", ") \
                           + str(tmp_rake[i]) + str(', ') \
                           + str(tmp_magnitude[i]) + str(", ") \
                           + str(tmp_misfit[i]) + str(", ") \
                           + str(tmp_segment[i]) + str(" \n")
                f.write(text_str)
        print("Searching result saved at {}.".format(output_file_path))
    except:
        print("Unable to write searching results at {}. \n".format(output_file_path))
        pass

    # write out results
    try:
        with open(result_file_path, "w") as f:
            f.write("Latitude, Longitude, Z, strike, dip, rake, magnitude, misifit, n_segment\n")
            text_str = str(result_lat) + str(", ") \
               + str(result_long) + str(", ") \
               + str(result_z) + str(", ") \
               + str(result_strike) + str(", ") \
               + str(result_dip) + str(", ") \
               + str(result_rake) + str(', ') \
               + str(result_magnitude) + str(", ") \
               + str(result_misfit) + str(", ") \
               + str(result_segment) + str(" \n")
            f.write(text_str)
    except:
        print("Uable to write results at {}. \n".format(result_file_path))
        pass

    # write out location log
    # write out the log
    try:
        with open(log_file_path, 'w') as f:
            f.write(str("* Source localization\n"))
            f.write(str("* Working parameters\n"))
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
            f.write("* Solver\n")
            f.write("Searching method: Grid search!\n")
            f.write("Using the pre-defined focal mechanism: ")
            f.write(str("Initial location (lat, long z): ({}, {}, {}). \n".format(np.round(initial_source[0], 4), np.round(initial_source[1], 4), np.round(initial_source[2], 4))))
            f.write(str("Strike = {}. \n".format(strike)))
            f.write(str("Dip = {}. \n".format(dip)))
            f.write(str("Rake = {}. \n".format(rake)))
            f.write(str("Magnitude = {}. \n".format(magnitude)))
            f.write(str("{} grid points are searched.\n".format(n_point)))
            f.write(str("Searching area: x=[{}, {}], y=[{}, {}], z=[{}, {}]".format(np.round(initial_source[0] - x_min_range, 6),
                                                                                    np.round(initial_source[0] + x_max_range, 6),
                                                                                    np.round(initial_source[1] - y_min_range, 6),
                                                                                    np.round(initial_source[1] + y_max_range, 6),
                                                                                    np.round(initial_source[2] * -1000 - z_min_range, 4),
                                                                                    np.round(initial_source[2] * -1000 + z_max_range, 4))))
            f.write(str("Maximum searching range = {} km.\n ".format(np.round(max_searching_distance, 4))))

    except:
        print("Unable to write log file at {}.".format(log_file_path))




def source_inversion(eventid, initial_source,
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
        print("*" * 10)
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
        print("*" * 10)
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

        # localization
        source_localization(dfs, initial_source, station_names, loc_sta_list)

    except:
        print("INVERSION ABORT UNEXPECTEDLY !")

    print("Time elapse = {} s. ".format(time.time() - t0))



if __name__ == '__main__':
    example_source_inversion()











