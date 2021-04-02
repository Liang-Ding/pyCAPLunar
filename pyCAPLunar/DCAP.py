# -------------------------------------------------------------------
# Moment Tensor Inversion.
#
# Ref:
# [1] TAPE Walter, TAPE Carl., 2015
# A uniform parametrization of moment tensors.
# GJI, 2015, 202(3): 2074–2081.
#
# [2] ZHU Lupei, HELMBERGER Donald V., 1996.
# Advancement in source estimation techniques using broadband regional seismograms.
# BSSA, 86(5): 1634–1641.
#
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from pyCAPLunar import g_TAPER_RATE_SRF, g_TAPER_SCALE
from pyCAPLunar.DSyn import DSynRotateRTZ, DENZ2RTZ
from pyCAPLunar.DPaste import DPaste

from obspy.geodetics.base import gps2dist_azimuth

import numpy as np


class DCAP():
    '''Moment tensor inversion using the Cut-And-Paste method.'''

    def __init__(self, sgts_list, data_list,
                 loc_src, loc_stations,
                 n_tp_sgt, n_ts_sgt,
                 n_tp_data, n_ts_data,
                 n_tp_max_shift, n_ts_max_shift,
                 dt, n_phase=5, n_component=3,
                 w_pnl=1.0, w_srf=1.0,
                 misfit_threshold=10,
                 reject_rate=0.1,
                 taper_scale=g_TAPER_SCALE,
                 taper_rate_srf=g_TAPER_RATE_SRF,
                 cc_threshold=0.4,
                 amplitude_ratio_threshold=2.0):

        self.dt = dt
        self.fn = 1.0/dt
        self.loc_src = np.zeros(3)
        self.loc_stations = loc_stations
        self.n_tp_sgt = np.asarray(n_tp_sgt).astype(int)
        self.n_ts_sgt = np.asarray(n_ts_sgt).astype(int)
        self.n_tp_data = np.asarray(n_tp_data).astype(int)
        self.n_ts_data = np.asarray(n_ts_data).astype(int)
        self.n_tp_max_shift = int(n_tp_max_shift)
        self.n_ts_max_shift = int(n_ts_max_shift)

        # constant variable
        self.n_phase = n_phase          # PNL in R and Z, and S/Surface in R, T and Z components.
        self.n_component = n_component  # 3, R-T-Z or E-N-Z component.
        self.n_station = len(sgts_list)
        self.n_segment = self.n_station * self.n_phase
        self.reject_rate = reject_rate
        self.misfit_threshold = misfit_threshold
        if np.fabs(cc_threshold) < 1:
            self.cc_threshold = np.fabs(cc_threshold)
        else:
            self.cc_threshold = np.fabs(cc_threshold) / 100.0
        self.MAX_MISFIT = 9999.0
        self.amp_ratio_threshold = amplitude_ratio_threshold

        # initial container
        self.azimuth_list = []
        self.distance_list = []
        self.sgt_pnl = []
        self.sgt_srf = []
        self.data_pnl_rtz = []
        self.data_srf_rtz = []

        # weighting factors
        self.w_pnl = w_pnl
        self.w_srf = w_srf

        # weight by distance, distance scale
        # Zhu & Helmberger, 1996
        self.w_pnl_distance_scale = 1.13
        self.w_Love_distance_scale = 0.55
        self.w_Rayleigh_distance_scale = 0.74


        # initialize.
        self.update_azimuth(loc_src)
        self.update_sgt(sgts_list)
        self.update_data(data_list)
        self.create_weights()

        # create the coefficient for accelerating the inversion.
        self.TAPER_SCALE = taper_scale  # should be LESS than 0.
        if taper_scale > 0:
            self.TAPER_SCALE = -1.0 * taper_scale
        # if taper_rate_srf = 1.0, then the s/surface will be discard.
        if taper_rate_srf < 0.75:
            self.taper_rate_srf = taper_rate_srf
        else:
            self.taper_rate_srf = 0.75

        self.taper_coeff = np.exp(self.TAPER_SCALE * np.arange(np.max(self.n_syn_lens_srf)))




    def update_azimuth(self, loc_src):
        '''Update azimuth and data component in R-T-Z system.'''
        if self.loc_src[0] == loc_src[0] and self.loc_src[1] == loc_src[1]:
            return
        else:
            self.loc_src = loc_src
            for i in range(self.n_station):
                dist, alpha12, alpha21 = gps2dist_azimuth(self.loc_src[0], self.loc_src[1],
                                                          self.loc_stations[i][0],
                                                          self.loc_stations[i][1])
                self.azimuth_list.append(np.deg2rad(alpha12))
                # distance in km.
                self.distance_list.append(np.sqrt(np.add(np.power(dist/1000, 2),
                                                         np.power(self.loc_src[2]-self.loc_stations[i][2], 2))))


    def update_sgt(self, sgts_list):
        '''Update sgt parameters once the SGT is reset.'''
        self.n_syn_lens_pnl = []
        self.n_syn_lens_srf = []
        for i in range(self.n_station):
            self.sgt_pnl.append(sgts_list[i][0])
            self.sgt_srf.append(sgts_list[i][1])
            # SGT shape: length, 3 , 6
            n_pnl_syn, _, _ = np.shape(sgts_list[i][0])
            n_srf_syn, _, _ = np.shape(sgts_list[i][1])
            self.n_syn_lens_pnl.append(n_pnl_syn)
            self.n_syn_lens_srf.append(n_srf_syn)


    def update_data(self, data_list):
        '''Update data. '''
        self.n_data_lens_pnl = []
        self.n_data_lens_srf = []
        for i in range(self.n_station):
            self.data_pnl_rtz.append(DENZ2RTZ(data_list[i][0], self.azimuth_list[i]))
            self.data_srf_rtz.append(DENZ2RTZ(data_list[i][1], self.azimuth_list[i]))
            # data shape: 3, length
            _, _pnl_length = np.shape(data_list[i][0])
            _, _srf_length = np.shape(data_list[i][1])
            self.n_data_lens_pnl.append(_pnl_length)
            self.n_data_lens_srf.append(_srf_length)


    def create_weights(self):
        '''
        Create weighting matrix.
        '''

        # initialize the weight matrix.
        self.weights = np.ones((self.n_station, self.n_phase))

        # weights by phase.
        self.weights[:, :2] = self.w_pnl * self.weights[:, :2]
        self.weights[:, 2:] = self.w_srf * self.weights[:, 2:]

        # weights by distance and phase type.
        # Parameters utilized for events in California are similar to Zhu & Helmberger, 1996.
        # [:, 0, 1] -> pnl, weight = $(r/r0)^{-1.13}$
        # [:, 2, 4] -> Rayleigh, weight = $(r/r0)^{-0.74}$
        # [:, 3] -> Love, weight= $(r/r0)^{-0.55}$

        r0 =100.0  # in km.
        for i in range(self.n_station):
            # PNL
            self.weights[i, 0] = self.weights[i, 0] * np.power(self.distance_list[i]/r0,
                                                               -1.0 * self.w_pnl_distance_scale)

            self.weights[i, 1] = self.weights[i, 1] * np.power(self.distance_list[i]/r0,
                                                               -1.0 * self.w_pnl_distance_scale)

            # Rayleigh
            self.weights[i, 2] = self.weights[i, 2] * np.power(self.distance_list[i] / r0,
                                                               -1.0 * self.w_Rayleigh_distance_scale)
            self.weights[i, 4] = self.weights[i, 4] * np.power(self.distance_list[i] / r0,
                                                               -1.0 * self.w_Rayleigh_distance_scale)

            # Love
            self.weights[i, 3] = self.weights[i, 3] * np.power(self.distance_list[i] / r0,
                                                               -1.0 * self.w_Love_distance_scale)



    def assess_misfit(self, misfit_matrix, cc_matrix):
        '''
        Calculate the weighting misfit.
        '''

        # generate the weights matrix.
        self.create_weights()

        n_reject_segment = 0
        sum_misfit = 0
        for i in range(self.n_station):
            for j in range(self.n_phase):
                if misfit_matrix[i][j] > self.misfit_threshold or cc_matrix[i][j] < self.cc_threshold:
                    n_reject_segment += 1
                    self.weights[i][j] = 0
                    continue
                else:
                    sum_misfit += self.weights[i][j] * misfit_matrix[i][j]
        # return the misfit according to the reject rate.
        if n_reject_segment/self.n_segment > self.reject_rate:
            return self.MAX_MISFIT
        else:
            return sum_misfit / (self.n_segment-n_reject_segment)


    def cut_and_paste(self, mt):
        '''
        Calculate the waveform misfit for the given moment tensor.
        '''
        misfit_matrix = np.ones((self.n_station, self.n_phase))
        shift_matrix = np.zeros((self.n_station, self.n_phase)).astype(int)
        cc_matrix = np.ones((self.n_station, self.n_phase))

        # synthetic wavefrom
        for i in range(self.n_station):
            # synthetic trace in R-T-Z.
            pnl_syn_rtz = DSynRotateRTZ(mt, self.sgt_pnl[i],
                                        self.azimuth_list[i], self.n_syn_lens_pnl[i])
            srf_syn_rtz = DSynRotateRTZ(mt, self.sgt_srf[i],
                                      self.azimuth_list[i], self.n_syn_lens_srf[i])

            # Taper
            for j in range(self.n_component):
                # PNL
                pnl_syn_rtz[j][:self.n_tp_sgt[i]] = pnl_syn_rtz[j][:self.n_tp_sgt[i]] * np.flip(self.taper_coeff[:self.n_tp_sgt[i]])
                # Surface
                # find the peak of the S/Surface after the S arrival.
                ts_peak = np.argmax(np.fabs(srf_syn_rtz[j, self.n_ts_sgt[i]:]))
                _taper_idx0 = self.n_ts_sgt[i] + int(ts_peak * self.taper_rate_srf)

                # taper the surface waveform at the left of the peak.
                srf_syn_rtz[j][:_taper_idx0] = srf_syn_rtz[j][:_taper_idx0] * np.flip(
                    self.taper_coeff[:_taper_idx0])
                # taper the surface waveform at the right of the peak.
                _taper_idx1 = self.n_ts_sgt[i] + int((2.0 - self.taper_rate_srf)*ts_peak)
                _length = len(srf_syn_rtz[j])
                if _taper_idx1 < _length:
                    srf_syn_rtz[j][_taper_idx1:] = srf_syn_rtz[j][_taper_idx1:] * self.taper_coeff[:_length - _taper_idx1]

            k = 0
            # pnl
            for j in range(self.n_component):
                if j == 1:
                    continue
                else:
                    # amplitude thresholding
                    _max_amp_data = np.max(np.fabs(self.data_pnl_rtz[i][j]))
                    _max_amp_syn = np.max(np.fabs(pnl_syn_rtz[j]))
                    if self.amp_ratio_threshold < _max_amp_data / _max_amp_syn or self.amp_ratio_threshold < _max_amp_syn / _max_amp_data:
                        misfit_matrix[i, k] = self.MAX_MISFIT
                        shift_matrix[i, k] = 0.0
                        cc_matrix[i, k] = 0.0
                    else:
                        # paste: shift and misfit calculation.
                        _n_shift, _misfit, _cc = DPaste(self.data_pnl_rtz[i][j],
                                                 self.n_tp_data[i],
                                                 self.n_data_lens_pnl[i],
                                                 pnl_syn_rtz[j],
                                                 self.n_tp_sgt[i],
                                                 self.n_syn_lens_pnl[i],
                                                 self.n_tp_max_shift)
                        misfit_matrix[i, k] = _misfit
                        shift_matrix[i, k] = _n_shift
                        cc_matrix[i, k] = _cc
                    k += 1
            # s/surface
            for j in range(self.n_component):
                # amplitude thresholding
                _max_amp_data = np.max(np.fabs(self.data_srf_rtz[i][j]))
                _max_amp_syn = np.max(np.fabs(srf_syn_rtz[j]))
                if self.amp_ratio_threshold < _max_amp_data / _max_amp_syn or self.amp_ratio_threshold < _max_amp_syn / _max_amp_data:
                    misfit_matrix[i, k] = self.MAX_MISFIT
                    shift_matrix[i, k] = 0.0
                    cc_matrix[i, k] = 0.0
                else:
                    # paste: shift and misfit.
                    _n_shift, _misfit, _cc = DPaste(self.data_srf_rtz[i][j],
                                             self.n_ts_data[i],
                                             self.n_data_lens_srf[i],
                                             srf_syn_rtz[j],
                                             self.n_ts_sgt[i],
                                             self.n_syn_lens_srf[i],
                                             self.n_ts_max_shift)
                    misfit_matrix[i, k] = _misfit
                    shift_matrix[i, k] = _n_shift
                    cc_matrix[i, k] = _cc
                k += 1
        return misfit_matrix, shift_matrix, cc_matrix


    def misfit(self, mt):
        '''Calculate the waveform misift. '''
        misfit_matrix, shift_matrix, cc_matrix = self.cut_and_paste(mt)
        return self.assess_misfit(misfit_matrix, cc_matrix)
