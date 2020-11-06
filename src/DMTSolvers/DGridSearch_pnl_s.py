# -------------------------------------------------------------------
# Moment tensor inversion based on the full waveform inversion.
# Grid search the Strike, Dip, and Rake.
# The pnl and surface are separate.
#
# The observation waveform is in R-T-Z system.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from DCAPs.DCAPSlice import DSlice_data
from DMTSolvers.DMisfit import DMisfit_pnl_s
from DMTSolvers.DFilter import DFilter_sgt, DFilter_data
import multiprocessing as mp
import numpy as np
import os

import time


def calculate(func, args):
    '''Function for calling and running.'''
    return func(*args)

def calculatestar(args):
    return calculate(*args)


def DFullgridSearch_pnl_s_MP(station_names, list_sgt_array, n_tp_array, n_ts_array,
                         list_obs_array, n_p_pick_array,
                         loc_src, loc_list_sta,
                         p_waveform_length, p_n_offset,
                         s_waveform_length, s_n_offset,
                         b_p_filter, p_freqmin, p_freqmax,
                         b_s_filter, s_freqmin, s_freqmax, df,
                         n_component, minimum_waveform_segment,
                         lag_threshold, cc_threshold, misfit_threshold,
                         strike_0, strike_1, d_strike,
                         dip_0, dip_1, d_dip,
                         rake_0, rake_1, d_rake,
                         mag_0, mag_1, d_mag,
                         PROCESSOR, log_saving_path=None):

    '''
    * Focal mechanism inversion by waveform fitting. 
    * Calculate the waveform misift by using grid search method.
    * 
    * Multiple processing (MP) method!
    *

    :param station_names:       The name list of all station.
                                Data type: List
                                Data shape: N
                                * N: The number of stations.
                                Data example: ["sta_1", "sta_2", ..., "sta_N"]

    :param list_sgt_array:      The sgt arrays of all stations.
                                * Size = N* [T, 3, 6].
                                * N: The number of stations.
                                * T: The number of time steps.
                                * 3: the number of unit force, constant.
                                * 6: The number of the elements in epsilon matrix, constant.

    :param n_tp_array:          The computing arrival time of P phase.
                                Data type: numpy 1d array.
                                Data shape: [N]
                                * N: The number of stations.

    :param n_ts_array:          The computing arrival time of s phase.
                                Data type: numpy 1d array.
                                Data shape: [N]
                                * N: The number of stations.

    :param list_obs_array:      The observing data (displacement) of all stations.
                                * Size = N * [3, n_sample]
                                * N: The number of stations.
                                * 3: The component of the displacement. constant
                                * n_sample: The number of sample.

    :param n_p_pick_array:      The picked arrival time of P phase.
                                Data type: numpy 1d array.
                                Data shape: N
                                * N: The number of stations.
    :param loc_src:
    :param loc_list_sta:
    :param p_waveform_length:   The waveform length for pnl. INT.
    :param p_n_offset:          The offset between the zero position of wave segment and the first break. INT.
    :param s_waveform_length:   The waveform length for surface.    INT
    :param s_n_offset:          The offset between the zero position of wave segment and the first break. INT
    :param b_p_filter:          Filter or not?  BOOL
    :param p_freqmin:           The low limit of the band pass filter for pnl wave. FLOAT
    :param p_freqmax:           The high limit of the band pass filter for pnl wave. FLOAT
    :param b_s_filter:          Filter or not?  BOOL
    :param s_freqmin:           The low limit of the and pass filter for the surface wave. FLOAT
    :param s_freqmax:           The high limit of the band pass filter for the surface wave.    FLOAT
    :param df:                  The sampling rate in Hz. FLOAT.

    :param n_component:         The number of component, in most of the case is 6, pnl and surface are separate.

    :param minimum_waveform_segment:    The minimum number of waveform segment that used to calculate the misfit.
                                        Data type: INT.

    :param lag_threshold:               The threshold of the shift (in sample), the maximum time lag (in sample)
                                        Data type: INT
    :param cc_threshold:                The minimum cross-correlation coefficient.
                                        Data type: float < 1.0 (eg: 0.8).

    :param misfit_threshold:            The maximum misfit_threshold.
                                        Data type: float

    :param strike_0:            The start of the strike range. float.
    :param strike_1:            The end of the strike range. float.
    :param d_strike:            The step of the strike. float.
    :param dip_0:               The start of the dip range. float.
    :param dip_1:               The end of the strike range. float.
    :param d_dip:               The step of the dip. float.
    :param rake_0:              The start of the rake range. float.
    :param rake_1:              The end of the rake. float.
    :param d_rake:              The step of the rake.
    :param mag_0:               The start of the magnitude range. float.
    :param mag_1:               The end of the magnitude range. float.
    :param d_mag:               The step of the magnitude. float.
    :param PROCESSOR:           The number of processor used.
    :param log_saving_path:     Saving the inversion process to TXT file.
    :return:                    The Focal Parameters and the magnitude.
    '''

    # checking data
    n_station = len(station_names)
    if n_station != len(list_sgt_array):
        print("Unbalanced station and SGT database.")
        return None, None, None, None

    if n_station != len(list_obs_array):
        print("Unbalanced station and recordings.")
        return None, None, None, None

    if log_saving_path is None:
        b_savelog = False
    else:
        # set true
        b_savelog = True
        # create file.
        log_saving_dir = os.path.dirname(log_saving_path)
        if not os.path.exists(log_saving_dir):
            os.makedirs(log_saving_dir, exist_ok=True)
        # Open the file to write.
        f_log = open(log_saving_path, 'w')

    # local variables - for return
    final_strike = 0
    final_dip = 0
    final_rake = 0
    final_magnitude = 0
    final_data_segment = 0
    final_lags = 0
    final_ccs = 0
    final_weights = 0
    final_wave_misift = 0
    final_syn_rtz_list = []
    final_data_rtz_list = []
    minimum_misfit = 999999.0
    max_data_segment = 0

    try:
        # Slice SGTs and waveform segments.
        sgts_list = []
        data_list = []
        for i_sta, (sgt_arr, data_arr) in enumerate(zip(list_sgt_array, list_obs_array)):

            ### SGT
            # pnl
            pnl_idx0_sgt = int(n_tp_array[i_sta] - p_n_offset)
            pnl_idx1_sgt = int(pnl_idx0_sgt + p_waveform_length)

            s_idx0_sgt = int(n_ts_array[i_sta] - s_n_offset)
            s_idx1_sgt = int(s_idx0_sgt + s_waveform_length)

            ## filter and slice
            # pnl
            if b_p_filter:
                p_sgt_arr = DFilter_sgt(sgt_arr, p_freqmin, p_freqmax, df)
                sgts_list.append(DSlice_data(p_sgt_arr, len(p_sgt_arr), pnl_idx0_sgt, pnl_idx1_sgt))
            else:
                sgts_list.append(DSlice_data(sgt_arr, len(sgt_arr), pnl_idx0_sgt, pnl_idx1_sgt))
            # s
            if b_s_filter:
                s_sgt_arr = DFilter_sgt(sgt_arr, s_freqmin, s_freqmax, df)
                sgts_list.append(DSlice_data(s_sgt_arr, len(s_sgt_arr), s_idx0_sgt, s_idx1_sgt))
            else:
                sgts_list.append(DSlice_data(sgt_arr, len(sgt_arr), s_idx0_sgt, s_idx1_sgt))

            ### DATA
            pnl_idx0_data = int(n_p_pick_array[i_sta] - p_n_offset)
            pnl_idx1_data = int(pnl_idx0_data + p_waveform_length)
            s_idx0_data = int(n_p_pick_array[i_sta] + (n_ts_array[i_sta] - n_tp_array[i_sta]) - s_n_offset)
            s_idx1_data = int(s_idx0_data + s_waveform_length)
            ## filter and slice
            # pnl
            if b_p_filter:
                p_data_arr = DFilter_data(data_arr, p_freqmin, p_freqmax, df)
                data_list.append(np.transpose(DSlice_data(np.transpose(p_data_arr), len(p_data_arr[0]), pnl_idx0_data, pnl_idx1_data)))
            else:
                data_list.append(np.transpose(DSlice_data(np.transpose(data_arr), len(data_arr[0]), pnl_idx0_data, pnl_idx1_data)))
            # s/surface
            if b_s_filter:
                s_data_arr = DFilter_data(data_arr, s_freqmin, s_freqmax, df)
                data_list.append(np.transpose(DSlice_data(np.transpose(s_data_arr), len(s_data_arr[0]), s_idx0_data, s_idx1_data)))
            else:
                data_list.append(np.transpose(DSlice_data(np.transpose(data_arr), len(data_arr[0]), s_idx0_data, s_idx1_data)))

        magnitude_array = np.arange(mag_0, mag_1, d_mag)
        strike_array = np.arange(strike_0, strike_1, d_strike)
        dip_array = np.arange(dip_0, dip_1, d_dip)
        rake_array = np.arange(rake_0, rake_1, d_rake)


        ######## parallel computing #########
        # Full grid search - search the strike, dip, rake, magnitude.
        # use multiprocessor pool.
        with mp.Pool(PROCESSOR) as pool:
            # parallel process - type 1, use unsorted map - fast??.
            TASKS = [(DMisfit_pnl_s, (strike, dip, rake, magnitude,
                                          sgts_list, data_list,
                                          loc_src, loc_list_sta,
                                          n_station, n_component,
                                          lag_threshold, cc_threshold,
                                          misfit_threshold,
                                          minimum_waveform_segment))
                     for strike in strike_array
                     for dip in dip_array
                     for rake in rake_array
                     for magnitude in magnitude_array]

            results = pool.imap_unordered(calculatestar, TASKS)
            for r in results:
                # write out searching results: strike, dip, rake, magnitude, misfit, n_valid_waveform_segment
                mean_wave_misfit = r[0]
                tmp_strike = r[1]
                tmp_dip = r[2]
                tmp_rake = r[3]
                tmp_magnitude = np.round(r[4], 6)
                tmp_data_segment = r[5]

                # write out focal parameters and waveform misfit.
                if b_savelog:
                    out_str = str(tmp_strike)+str(", ")+str(tmp_dip)+str(", ")\
                          +str(tmp_rake)+str(", ")+str(tmp_magnitude)+str(", ")\
                          +str(mean_wave_misfit) + str(", ")+str(int(tmp_data_segment))

                    f_log.write(str(out_str)+str('\n'))

                # save the parameters to the data container if meet the requirements
                if 1.0/minimum_misfit * max_data_segment < 1.0/mean_wave_misfit * tmp_data_segment:
                    minimum_misfit = mean_wave_misfit
                    max_data_segment = tmp_data_segment
                    final_strike = r[1]
                    final_dip = r[2]
                    final_rake = r[3]
                    final_magnitude = np.round(r[4], 2)
                    final_data_segment = r[5]
                    final_lags = r[6]
                    final_ccs = r[7]
                    final_weights = r[8]
                    final_wave_misift = r[9]
                    final_syn_rtz_list = r[10]
                    final_data_rtz_list = r[11]

        if b_savelog:
            f_log.write('\n')

    # if program abort.
    except:
        if b_savelog:
            f_log.write(str("Grid search process abort!"))
            f_log.close()
        print("Inversion abort with the error -1!")
        exit(-1)

    # close file.
    if b_savelog:
        f_log.close()

    return final_strike, final_dip, final_rake, final_magnitude, minimum_misfit, final_data_segment, \
           final_lags, final_ccs, final_weights, final_wave_misift, final_syn_rtz_list, final_data_rtz_list

































