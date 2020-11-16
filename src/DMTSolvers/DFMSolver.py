# -------------------------------------------------------------------
# Focal mechanism inversion.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from DMTSolvers import DL_ERROR_MISIFT

from DCAPs.DCAPShift import DShift_1d
from DCAPs.DCAPSlice import DSlice_data
from DCAPs.DCAPExtend import DExtend_sub
from DCAPs.DCAPUtils import momentMagnitude_2_moment, taper
from Seismology.DFilter import bandpass
from Seismology.geo_tools import gps2dist_azimuth
from Seismology.DFocalTools import DFocalPlane_2_mt
from Seismology.Rotating_tools import DRotate_ENZ2RTZ
from DSynthetics.SGT2Disp import SGTs_2_Displacement

import os
from datetime import datetime
import time
import numpy as np


class DFMGridSearch():
    '''Focal Mechanism inversion based on the GridSearch method.'''

    def __init__(self, sgt_array_list, n_computed_tp_array, n_computed_ts_array,
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
                 b_write_best_result_file=True):

        self.project_dir = project_dir
        # Linux
        if str(self.project_dir)[-1] != str('/'):
            self.project_dir = str(self.project_dir) + str('/') + str("grid_search/")
        else
            self.project_dir = str(self.project_dir) + str("grid_search/")

        # save the searching result for each step.
        self.__job_identifier = job_identifier
        self.b_write_each_step = b_write_each_step
        self.b_write_working_log = b_write_working_log
        self.b_write_best_result_file = b_write_best_result_file
        self.create_file_path()

        # sgt array and data.
        self.data_array_list = data_array_list
        self.__sgt_array_list = sgt_array_list
        self.n_station = len(self.data_array_list)

        # arrival time.
        self.n_computed_tp_array = n_computed_tp_array  # tp for synthetic in sample. INT
        self.n_computed_ts_array = n_computed_ts_array  # ts for the synthetic in sample. INT
        self.n_picked_tp_array = n_picked_tp_array  # picked tp on the data in sample. INT

        # data cutting parameters
        self.n_pnl_length = n_pnl_length  # the length of PNL in sample. INT.
        self.n_surface_length = n_surface_length  # the lengh of S/Surface in sample. INT.
        self.n_pnl_back_offset = n_pnl_back_offset  # the gap between the waveform begin and the first P in sample. INT.
        self.n_surface_back_offset = n_surface_back_offset  # the gap between the waveform begin and the first S in sample. INT.

        self.loc_sta_list = loc_sta_list  # the location of each station.
        self.__loc_src = loc_src  # the initial source location to rotate the waveform.
        self.n_component = 6  # the Pnl and the S/Surface in R-T-Z.

        # searching parameters
        self.strike_0 = strike_0
        self.strike_1 = strike_1
        self.d_strike = d_strike

        self.dip_0 = dip_0
        self.dip_1 = dip_1
        self.d_dip = d_dip

        self.rake_0 = rake_0
        self.rake_1 = rake_1
        self.d_rake = d_rake

        self.mag_0 = mag_0
        self.mag_1 = mag_1
        self.d_mag = d_mag

        self.__strike_range = np.arange(self.strike_0, self.strike_1, self.d_strike)
        self.__dip_range = np.arange(self.dip_0, self.dip_1, self.d_dip)
        self.__rake_range = np.arange(self.rake_0, self.rake_1, self.d_rake)
        self.__mag_range = np.arange(self.mag_0, self.mag_1, self.d_mag)

        # threshold
        self.misfit_threshold = misfit_threshold
        self.timeshift_threshold = timeshift_threshold
        self.cc_threshold = cc_threshold
        self.n_valid_waveform_threshold = n_valid_waveform_threshold  # number of valid waveform segment.

        # filter
        self.b_pnl_filter = b_pnl_filter
        self.b_surface_filter = b_surface_filter
        self.pnl_freq_min = pnl_freq_min
        self.pnl_freq_max = pnl_freq_max
        self.surface_freq_min = surface_freq_min
        self.surface_freq_max = surface_freq_max

        # The sampling rate in Hz. the SGT and the observation data must have the same df.
        self.df = df

        # CONSTANT parameters for the SGT database
        self.n_force = 3  # [N, E, Z(up)], n_dim
        self.n_paras = 6  # [Mxx, Myy, Mzz, Mxy, Mxz, Myz].

        # container to save the cut sgt and the data segment.
        self.sgts_list = []
        self.data_list = []

        # rotate the data and save as global variable.
        self.pnl_data_rtz_list = []
        self.surface_data_rtz_list = []

        # container to save the lag, cc, misfit, weight
        self.waveform_lag_array = np.zeros([self.n_station, self.n_component])
        self.waveform_cc_array = np.zeros([self.n_station, self.n_component])
        self.waveform_misift_array = np.zeros([self.n_station, self.n_component])
        self.waveform_weight_array = np.ones([self.n_station, self.n_component])
        self.rotation_angle = np.zeros(self.n_station)

        # parameters for estimate the solution.
        self.strike = 0
        self.dip = 0
        self.rake = 0
        self.magnitude = 0
        self.reset_current_misfit()

        # parameters for the check().
        self.data_compatibility = True
        self.b_cut_waveform = False
        self.b_prepared = False
        self.b_rotate_date = True
        self.b_cut_sgt = True
        self.b_cut_data = True
        self.b_run_gridsearch = True

        self.t0 = 0
        self.time_elapse = 0

    def reset(self, sgt_array_list, n_computed_tp_array, n_computed_ts_array,
              data_array_list, n_picked_tp_array, loc_src, loc_sta_list,
              n_pnl_length, n_pnl_back_offset, n_surface_length, n_surface_back_offset,
              b_pnl_filter, pnl_freq_min, pnl_freq_max,
              b_surface_filter, surface_freq_min, surface_freq_max, df,
              misfit_threshold, timeshift_threshold, cc_threshold, n_valid_waveform_threshold,
              strike_0, strike_1, d_strike,
              dip_0, dip_1, d_dip,
              rake_0, rake_1, d_rake,
              mag_0, mag_1, d_mag,
              project_dir, job_identifier, b_write_each_step=True, b_write_working_log=True):

        '''Initialize and reset the parameters. '''
        self.__init__(sgt_array_list, n_computed_tp_array, n_computed_ts_array,
                      data_array_list, n_picked_tp_array, loc_src, loc_sta_list,
                      n_pnl_length, n_pnl_back_offset, n_surface_length, n_surface_back_offset,
                      b_pnl_filter, pnl_freq_min, pnl_freq_max,
                      b_surface_filter, surface_freq_min, surface_freq_max, df,
                      misfit_threshold, timeshift_threshold, cc_threshold, n_valid_waveform_threshold,
                      strike_0, strike_1, d_strike,
                      dip_0, dip_1, d_dip,
                      rake_0, rake_1, d_rake,
                      mag_0, mag_1, d_mag,
                      project_dir, job_identifier, b_write_each_step, b_write_working_log)

    def __write_log_header(self, fw):
        n_star = 8
        text_str = str('*' * n_star) + str(' pySI_3DSGT ') + str('*' * n_star) + str('\n\n')
        fw.write(text_str)
        text_str = str('*' * n_star) + str(' working log ') + str('*' * n_star) + str('\n\n')
        fw.write(text_str)

    def abort(self):
        try:
            with open(self.b_write_working_log, 'w') as fw:
                self.__write_log_header(fw)

                if not self.data_compatibility:
                    fw.write("The input data is incompatible. \n Please check the input data.\n")
        except:
            exit(-1)

    def create_file_path(self):
        '''Create folder and file path.'''
        self.save_step_dir = str(self.project_dir)
        self.save_step_file_path = str(self.save_step_dir) + str("job_") + str(self.__job_identifier) + str(".txt")
        self.best_result_file_path = str(self.save_step_dir) + str("job_") + str(self.__job_identifier) + str(
            "_best.txt")
        self.log_file_path = str(self.save_step_dir) + str("job_") + str(self.__job_identifier) + str("_log.txt")

    def check(self):
        '''Check the folder and eligibility.'''
        if not os.path.exists(self.project_dir):
            os.makedirs(self.project_dir, exist_ok=True)

        if not os.path.exists(self.save_step_dir):
            os.makedirs(self.save_step_dir, exist_ok=True)

        if self.n_station != len(self.data_array_list) or \
                self.n_station != len(self.n_computed_tp_array) or \
                self.n_station != len(self.n_computed_ts_array) or \
                self.n_station != len(self.n_picked_tp_array) or \
                self.n_station != len(self.loc_sta_list):
            self.data_compatibility = False
            self.abort()
        self.b_prepared = True

    def filter_sgt(self, sgt_array, freqmin, freqmax, df):
        ''' SGT filter'''
        new_sgt_array = np.zeros_like(sgt_array)
        for i in range(self.n_force):
            for j in range(self.n_paras):
                new_sgt_array[:, i, j] = bandpass(sgt_array[:, i, j], freqmin=freqmin, freqmax=freqmax,
                                                  df=df, corners=4, zerophase=False)
        return new_sgt_array

    def filter_data(self, data_arr, freqmin, freqmax, df):
        '''Observation data filter. '''
        new_data_arr = np.zeros_like(data_arr)
        for i, tr in enumerate(data_arr):
            new_data_arr[i] = bandpass(tr, freqmin=freqmin, freqmax=freqmax,
                                       df=df, corners=4, zerophase=False)
        return new_data_arr

    def cut_sgt(self):
        self.sgts_list = []
        ''' Cut the sgt into pnl and s/surface segments. '''
        for i_sta, (sgt_arr, data_arr) in enumerate(zip(self.__sgt_array_list, self.data_array_list)):
            ''' Cut SGT'''
            # pnl - the cutting range
            pnl_idx0_sgt = int(self.n_computed_tp_array[i_sta] - self.n_pnl_back_offset)
            pnl_idx1_sgt = int(pnl_idx0_sgt + self.n_pnl_length)
            # S/surface - the cutting range
            s_idx0_sgt = int(self.n_computed_ts_array[i_sta] - self.n_surface_back_offset)
            s_idx1_sgt = int(s_idx0_sgt + self.n_surface_length)
            ## filter and slice
            # pnl
            if self.b_pnl_filter:
                p_sgt_arr = self.filter_sgt(sgt_arr, self.pnl_freq_min, self.pnl_freq_max, self.df)
                self.sgts_list.append(DSlice_data(p_sgt_arr, len(p_sgt_arr), pnl_idx0_sgt, pnl_idx1_sgt))
            else:
                self.sgts_list.append(DSlice_data(sgt_arr, len(sgt_arr), pnl_idx0_sgt, pnl_idx1_sgt))
            # S/Surface
            if self.b_surface_filter:
                s_sgt_arr = self.filter_sgt(sgt_arr, self.surface_freq_min, self.surface_freq_max, self.df)
                self.sgts_list.append(DSlice_data(s_sgt_arr, len(s_sgt_arr), s_idx0_sgt, s_idx1_sgt))
            else:
                self.sgts_list.append(DSlice_data(sgt_arr, len(sgt_arr), s_idx0_sgt, s_idx1_sgt))

        self.b_cut_sgt = False

    def cut_data(self):
        ''' Cut DATA into PNL and S/Surface segments. '''
        self.data_list = []
        for i_sta, (sgt_arr, data_arr) in enumerate(zip(self.__sgt_array_list, self.data_array_list)):
            # pnl - cut range
            pnl_idx0_data = int(self.n_picked_tp_array[i_sta] - self.n_pnl_back_offset)
            pnl_idx1_data = int(pnl_idx0_data + self.n_pnl_length)
            # S/surface - cut range
            s_idx0_data = int(self.n_picked_tp_array[i_sta] + (
                    self.n_computed_ts_array[i_sta] - self.n_computed_tp_array[i_sta]) - self.n_surface_back_offset)
            s_idx1_data = int(s_idx0_data + self.n_surface_length)

            ## filter and slice
            # pnl
            if self.b_pnl_filter:
                p_data_arr = self.filter_data(data_arr, self.pnl_freq_min, self.pnl_freq_max, self.df)
                self.data_list.append(np.transpose(
                    DSlice_data(np.transpose(p_data_arr), len(p_data_arr[0]), pnl_idx0_data, pnl_idx1_data)))
            else:
                self.data_list.append(np.transpose(
                    DSlice_data(np.transpose(data_arr), len(data_arr[0]), pnl_idx0_data, pnl_idx1_data)))
            # s/surface
            if self.b_surface_filter:
                s_data_arr = self.filter_data(data_arr, self.surface_freq_min, self.surface_freq_max, self.df)
                self.data_list.append(np.transpose(
                    DSlice_data(np.transpose(s_data_arr), len(s_data_arr[0]), s_idx0_data, s_idx1_data)))
            else:
                self.data_list.append(
                    np.transpose(DSlice_data(np.transpose(data_arr), len(data_arr[0]), s_idx0_data, s_idx1_data)))

        self.b_cut_data = False


    def set_strike_range(self, data_array):
        '''Manually set the strike searching range. '''
        if 0 == len(data_array):
            return False
        else:
            self.__strike_range = data_array
            self.b_run_gridsearch = True
            return True

    def set_dip_range(self, data_array):
        '''Manually set the dip searching range. '''
        if 0 == len(data_array):
            return False
        else:
            self.__dip_range = data_array
            self.b_run_gridsearch = True

    def set_rake_range(self, data_array):
        '''Manually set the rake searching range. '''
        if 0 == len(data_array):
            return False
        else:
            self.__rake_range = data_array
            self.b_run_gridsearch = True

    def set_magnitude_range(self, data_array):
        '''Manually set the rake searching range. '''
        if 0 == len(data_array):
            return False
        else:
            self.__mag_range = data_array
            self.b_run_gridsearch = True

    def set_job_identifier(self, job_identifier):
        '''Set job identifier and create file path.'''
        self.__job_identifier = job_identifier
        self.create_file_path()

    def set_Strain_Greens_Function(self, sgt_array_list):
        '''Set new strain Green's Function. '''
        self.__sgt_array_list = sgt_array_list
        self.b_cut_sgt = True
        self.b_rotate_date = True

    def set_source_location(self, source_location):
        ''' Set new source location. '''
        self.__loc_src = source_location
        self.b_rotate_date = True

    def set_computed_tp_array(self, n_computed_tp_array):
        self.n_computed_tp_array = n_computed_tp_array  # tp for synthetic in sample. INT
        self.b_cut_sgt = True
        self.b_cut_data = True

    def set_computed_ts_array(self, n_computed_ts_array):
        self.n_computed_ts_array = n_computed_ts_array  # ts for the synthetic in sample. INT
        self.b_cut_sgt = True
        self.b_cut_data = True

    def reset_current_misfit(self):
        '''Reset the default misift and valid waveform segment for other usage. '''
        self.current_misift = DL_ERROR_MISIFT
        self.current_valid_waveform = 0

    def set_write_searching_step(self, bool_value):
        '''Set whether to save the result of all searching steps.'''
        self.b_write_each_step = bool_value

    def set_write_working_log(self, bool_value):
        '''Set whether to save the working log.'''
        self.b_write_working_log = bool_value

    def set_write_best_result(self, bool_value):
        '''Set whether to write out best result to file. '''
        self.b_write_best_result_file = bool_value

    def misfit(self, strike, dip, rake, magnitude):
        '''
        ! Calculate the waveform misfit between the synthetic and the data.

        :param strike:      The strike.     DEGREE.
        :param dip:         The dip.        DEGREE.
        :param rake:        The rake.       DEGREE
        :param magnitude:   The magnitude.  float.
        :return:
        '''

        # generate moment tensor
        moment = momentMagnitude_2_moment(np.round(magnitude, 2))
        mt = DFocalPlane_2_mt(np.deg2rad(strike), np.deg2rad(dip), np.deg2rad(rake), m0=moment)

        # container to save the lag, cc, misfit, weight
        waveform_lag_array = np.zeros([self.n_station, self.n_component])
        waveform_cc_array = np.zeros([self.n_station, self.n_component])
        waveform_misift_array = np.zeros([self.n_station, self.n_component])
        waveform_weight_array = np.ones([self.n_station, self.n_component])

        for i_sta in range(self.n_station):
            # the container to store the pnl and s/surface segment.
            syn_trace_list = []
            data_trace_list = []

            # synthetic waveform in ENZ
            pnl_syn_enz = SGTs_2_Displacement(mt, self.sgts_list[i_sta * 2])
            s_syn_enz = SGTs_2_Displacement(mt, self.sgts_list[i_sta * 2 + 1])
            # synthetic waveform in rtz
            pnl_syn_rtz = DRotate_ENZ2RTZ(pnl_syn_enz, ba=self.rotation_angle[i_sta])
            s_syn_rtz = DRotate_ENZ2RTZ(s_syn_enz, ba=self.rotation_angle[i_sta])

            # data in rtz
            pnl_data_rtz = self.pnl_data_rtz_list[i_sta]
            s_data_rtz = self.surface_data_rtz_list[i_sta]

            # restore and merge all 6 traces in one list.
            # syn - pnl, s
            for tr in pnl_syn_rtz:
                syn_trace_list.append(tr)
            for tr in s_syn_rtz:
                syn_trace_list.append(tr)

            # data - pnl, s
            for tr in pnl_data_rtz:
                data_trace_list.append(tr)
            for tr in s_data_rtz:
                data_trace_list.append(tr)

            # The weights for the PNL in the transverse component are set to 0.
            for i_comp in np.arange(self.n_component):
                if i_comp == 1:
                    waveform_weight_array[i_sta][i_comp] = 0
                    continue

                # taper
                syn_trace_list[i_comp] = taper(syn_trace_list[i_comp])
                data_trace_list[i_comp] = taper(data_trace_list[i_comp])

                # Compute the lag and CC for each component individually.
                lag, cc = DShift_1d(syn_trace_list[i_comp], data_trace_list[i_comp])

                waveform_lag_array[i_sta][i_comp] = lag / self.df
                waveform_cc_array[i_sta][i_comp] = cc

                # extend the syn and data by add zero
                ext_syn_tr = DExtend_sub(syn_trace_list[i_comp], lag)
                ext_data_tr = DExtend_sub(data_trace_list[i_comp], -1 * lag)

                # Misfit function:
                # Zhu, L., & Helmberger, D. V. (1996). Advancement in source estimation. BSSA, 86(5), 1634–1641
                var_misfit = np.sum(np.square(np.subtract(ext_syn_tr, ext_data_tr))) / np.dot(np.fabs(ext_syn_tr),
                                                                                              np.fabs(ext_data_tr))
                waveform_misift_array[i_sta][i_comp] = var_misfit

                # weighting factor
                # if the waveform misfit is larger than the threshold, the weight is set to 0.
                if var_misfit > self.misfit_threshold:
                    waveform_weight_array[i_sta][i_comp] = 0
                else:
                    if lag > self.timeshift_threshold:
                        waveform_weight_array[i_sta][i_comp] = 0
                    elif cc < self.cc_threshold:
                        waveform_weight_array[i_sta][i_comp] = 0

        # count valid waveform segment
        n_valid_waveform = np.count_nonzero(waveform_weight_array)

        if n_valid_waveform < self.n_valid_waveform_threshold or n_valid_waveform == 0:
            return DL_ERROR_MISIFT, n_valid_waveform, waveform_lag_array, waveform_cc_array, \
                   waveform_misift_array, waveform_weight_array, syn_trace_list, data_trace_list
        else:
            # compute mean waveform misfit
            return np.mean(np.hstack(waveform_misift_array[np.nonzero(waveform_weight_array)])), \
                   n_valid_waveform, waveform_lag_array, waveform_cc_array, \
                   waveform_misift_array, waveform_weight_array, syn_trace_list, data_trace_list

    def evaluate_solution(self, misfit, n_valid_waveform):
        '''
        * Evaluate the current solution.

        :param misfit:              The average waveform misfit.
        :param n_valid_waveform:    The number of valid waveform.
        :return: Ture (Accept the new solution) / False (Reject).
        '''

        if 1.0 / self.current_misift * self.current_valid_waveform < 1.0 / misfit * n_valid_waveform:
            self.current_misift = misfit
            self.current_valid_waveform = n_valid_waveform
            return True
        else:
            return False

    def run_gridsearch(self):
        '''grid search'''

        # check status.
        if not self.b_prepared:
            self.check()

        if self.b_cut_sgt:
            self.cut_sgt()

        if self.b_cut_data:
            self.cut_data()

        self.t0 = datetime.now()
        t0 = time.time()

        if self.b_rotate_date:
            self.pnl_data_rtz_list = []
            self.surface_data_rtz_list = []

            for i_sta in range(self.n_station):
                pnl_data_enz = self.data_list[i_sta * 2]
                s_data_enz = self.data_list[i_sta * 2 + 1]

                dist, self.rotation_angle[i_sta], _ = gps2dist_azimuth(self.__loc_src[0], self.__loc_src[1],
                                                                       self.loc_sta_list[i_sta][0],
                                                                       self.loc_sta_list[i_sta][1])

                self.pnl_data_rtz_list.append(DRotate_ENZ2RTZ(pnl_data_enz, ba=self.rotation_angle[i_sta]))
                self.surface_data_rtz_list.append(DRotate_ENZ2RTZ(s_data_enz, ba=self.rotation_angle[i_sta]))
                self.b_rotate_date = False

        # data container
        n_record = len(self.__strike_range) * len(self.__dip_range) * len(self.__rake_range) * len(self.__mag_range)
        if n_record <= 0:
            return False

        record_mean_misfit = np.zeros(n_record)
        record_n_valid_waveform = np.zeros(n_record)
        record_strike = np.zeros(n_record)
        record_dip = np.zeros(n_record)
        record_rake = np.zeros(n_record)
        record_mag = np.zeros(n_record)
        i_record = 0
        for strike in self.__strike_range:
            for dip in self.__dip_range:
                for rake in self.__rake_range:
                    for magnitude in self.__mag_range:
                        # print("strike=", strike, ", dip=", dip, ", rake=", rake)

                        mean_waveform_misfit, n_valid_waveform, waveform_lag_array, \
                        waveform_cc_array, waveform_misift_array, waveform_weight_array, \
                        syn_trace_list, data_trace_list = self.misfit(strike, dip, rake, magnitude)

                        record_mean_misfit[i_record] = mean_waveform_misfit
                        record_n_valid_waveform[i_record] = n_valid_waveform
                        record_strike[i_record] = strike
                        record_dip[i_record] = dip
                        record_rake[i_record] = rake
                        record_mag[i_record] = magnitude

                        i_record += 1
                        if DL_ERROR_MISIFT == mean_waveform_misfit:
                            continue

                        # evaluate the current solution
                        if self.evaluate_solution(mean_waveform_misfit, n_valid_waveform):
                            self.strike = strike
                            self.dip = dip
                            self.rake = rake
                            self.magnitude = magnitude
                            self.waveform_lag_array = waveform_lag_array
                            self.waveform_cc_array = waveform_cc_array
                            self.waveform_misift_array = waveform_misift_array
                            self.waveform_weight_array = waveform_weight_array

        self.time_elapse = time.time() - t0

        if self.b_write_working_log:
            self.write_log_file()

        # write out each step
        if self.b_write_each_step:
            with open(self.save_step_file_path, 'w') as f:
                n_record = len(record_strike)
                for i in range(n_record):
                    strike = record_strike[i]
                    dip = record_dip[i]
                    rake = record_rake[i]
                    magnitude = record_mag[i]
                    text_str = str(strike) + str(", ") + str(dip) + str(", ") \
                               + str(rake) + str(", ") + str(magnitude) + str(', ') \
                               + str(record_mean_misfit[i]) + str(', ') + \
                               str(record_n_valid_waveform[i]) + str('\n')
                    f.write(text_str)

        if self.b_write_best_result_file:
            self.write_best_result_file()

        self.b_run_gridsearch = False
        return True

    def write_log_file(self):
        '''Write out parameters'''
        try:
            with open(self.log_file_path, 'w') as f:
                self.__write_log_header(f)

                f.write(str("* Working parameters\n"))
                f.write(str("Project directory:") + str(self.project_dir) + str('\n'))
                f.write(str("Job idetifier: ") + str(self.__job_identifier) + str('\n'))
                if self.b_write_each_step:
                    f.write(str("Step results saved at {}. \n ".format(self.save_step_file_path)))

                f.write(str("Best solution saved at {}. \n".format(self.best_result_file_path)))
                f.write(str("Solver started at {}. \n".format(str(self.t0))))
                f.write(str("Time cost = {} s. \n".format(self.time_elapse)))

                f.write("\n")
                f.write(str("* data\n"))
                f.write(str("The Solver using the Strain Green's function database.\n"))
                f.write(str("{} stations are used. \n".format(self.n_station)))
                f.write(str("Sampling interval = {} Hz.\n".format(self.df)))
                f.write(str("PNL offset = {} s. \n ".format(np.round(self.n_pnl_back_offset / self.df, 4))))
                f.write(str("PNL length = {} s. \n ".format(np.round(self.n_pnl_length / self.df, 4))))

                f.write(str("S/Surface offset = {} s. \n ".format(np.round(self.n_surface_back_offset / self.df, 4))))
                f.write(str("S/Surface length = {} s. \n ".format(np.round(self.n_surface_length / self.df, 4))))

                f.write("\n")
                f.write(str("* FILTERS\n"))
                f.write("Filter type: Band-pass\n")
                f.write("PNL: {} - {} Hz\n".format(self.pnl_freq_min, self.pnl_freq_max))
                f.write("S/Surface: {} - {} Hz\n".format(self.surface_freq_min, self.surface_freq_max))

                f.write("\n")
                f.write(str("* THRESHOLD\n"))
                f.write(str("Threshold of CC = ") + str(self.cc_threshold) + str(". \n"))
                f.write(str("Threshold of time shift = ") + str(np.round(self.timeshift_threshold, 4)) + str(" s. \n"))
                f.write(str("Threshold of misift = ") + str(np.round(self.misfit_threshold, 4)) + str(". \n"))
                f.write(
                    str("The minimum number of valid waveform = ") + str(self.n_valid_waveform_threshold) + str(". \n"))

                f.write("\n")
                f.write("* Solver\n")
                f.write("Searching method: Grid search!\n")
                f.write(str("Strike: [{}, {}], step: {}. \n".format(self.strike_0, self.strike_1, self.d_strike)))
                f.write(str("Dip: [{}, {}], step: {}. \n".format(self.dip_0, self.dip_1, self.d_dip)))
                f.write(str("Rake: [{}, {}], step: {}. \n".format(self.rake_0, self.rake_1, self.d_rake)))
                f.write(str("Magnitude: [{}, {}], step: {}. \n".format(self.mag_0, self.mag_1, self.d_mag)))
        except:
            exit(-1)

    def write_best_result_file(self):
        try:
            with open(self.best_result_file_path, "w") as f:
                f.write("strike, dip, rake, misift, number_of_valid_waveform_segment\n")
                text_str = str(self.strike) + str(', ') + str(self.dip) + str(', ') \
                           + str(self.rake) + str(', ') + str(self.magnitude) + str(', ') \
                           + str(self.current_misift) + str(', ') + str(self.current_valid_waveform) + str('\n')
                f.write(text_str)

                f.write("\n*Time shift\n")
                for lag in self.waveform_lag_array:
                    for l in lag:
                        f.write(str(l) + str(', '))
                    f.write('\n')

                f.write("\n*Cross-correlation Coefficient\n")
                for cc in self.waveform_cc_array:
                    for c in cc:
                        f.write(str(c) + str(', '))
                    f.write('\n')

                f.write("\n*Misfit\n")
                for misfit in self.waveform_misift_array:
                    for m in misfit:
                        f.write(str(m) + str(', '))
                    f.write('\n')

                f.write("\n*Weight\n")
                for weight in self.waveform_weight_array:
                    for w in weight:
                        f.write(str(w) + str(', '))
                    f.write('\n')
        except:
            pass

    def get_focal_mechanism(self):
        '''Enquire the focal mechanism.'''
        if self.b_run_gridsearch:
            self.run_gridsearch()
        return self.strike, self.dip, self.rake, self.magnitude, self.current_misift, self.current_valid_waveform

