# -------------------------------------------------------------------
# Moment tensor searching tools.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from MTTools.DMomentTensors import DMT_enz
from pyCAPLunar.DCAPUtils import mag2moment
from pyCAPLunar.DSyn import DSynRotateRTZ
from pyCAPLunar.DCAP import DCAP
import numpy as np


class DGridSearch(DCAP):
    def __init__(self, sgts_list, data_list,
                 loc_src, loc_stations,
                 n_tp_sgt, n_ts_sgt,
                 n_tp_data, n_ts_data,
                 n_tp_max_shift, n_ts_max_shift,
                 dt, n_phase=5, n_component=3,
                 w_pnl=1.0, w_srf=1.0,
                 misfit_threshold=10,
                 reject_rate=0.1,
                 taper_scale=-0.5, taper_rate_srf=0.3,
                 cc_threshold=0.4):

        super().__init__(sgts_list, data_list,
                 loc_src, loc_stations,
                 n_tp_sgt, n_ts_sgt,
                 n_tp_data, n_ts_data,
                 n_tp_max_shift, n_ts_max_shift,
                 dt, n_phase, n_component,
                 w_pnl, w_srf,
                 misfit_threshold,
                 reject_rate,
                 taper_scale, taper_rate_srf,
                 cc_threshold)


        self.NO_MAG = 999
        self.mag0 = self.NO_MAG
        self.mag1 = self.NO_MAG
        self.search_times_mag = 20   # default.

        # results
        self.strike = 0
        self.dip = 0
        self.rake = 0
        self.magnitude = 0
        # DC by default.
        self.lune_lat = 90
        self.lune_long = 0
        self.min_misfit = self.MAX_MISFIT

        # initialize the magnitude.
        while(self.mag0 == self.NO_MAG or self.mag1 == self.NO_MAG):
            self.init_magnitude()


    def init_magnitude(self):
        '''initialize the magnitude range by random moment tensor frame (Strike-dip-rake). '''

        self.check_magnitude(np.arange(0, 10, 1))
        mag0_arr = []
        mag1_arr = []

        if self.mag0 != self.NO_MAG and self.mag1 != self.NO_MAG:
            self.dmag = (self.mag1 - self.mag0) / 10

            mag0 = self.mag0
            mag1 = self.mag1
            for i in range(self.search_times_mag):
                self.check_magnitude(np.arange(mag0, mag1 + self.dmag, self.dmag))
                mag0_arr.append(self.mag0)
                mag1_arr.append(self.mag1)
        self.mag0 = np.min(mag0_arr)
        self.mag1 = np.max(mag1_arr)


    def check_magnitude(self, magnitude_array):
        '''Use constant strike-dp-rake to verify the magnitude.'''
        # random frame.
        strike = np.deg2rad(int(np.random.rand(1)*360))
        dip = np.deg2rad(int(np.random.rand(1)*90))
        rake = np.deg2rad(int(np.random.rand(1)*90-180))

        # in double coupe
        lune_lat_rad = np.deg2rad(90)
        lune_long_rad = np.deg2rad(0)

        difference_array = []
        n_count = len(magnitude_array)
        for magnitude in magnitude_array:
            m0 = mag2moment(magnitude)
            sum_difference = 0
            mt = m0 * DMT_enz(strike, dip, rake, lune_lat_rad, lune_long_rad)
            for i in range(self.n_station):
                srf_syn_rtz = DSynRotateRTZ(mt, self.sgt_srf[i],
                                            self.azimuth_list[i], self.n_syn_lens_srf[i])
                for j in range(self.n_component):
                    sum_difference += (max(abs(srf_syn_rtz[j])) - max(abs(self.data_srf_rtz[i][j])))
            difference_array.append(sum_difference)

        # select the best range.
        for i in range(n_count-1):
            if (difference_array[i] * difference_array[i+1]) < 0:
                self.mag0 = magnitude_array[i]
                self.mag1 = magnitude_array[i+1]
                break
            else:
                self.mag0 = self.NO_MAG
                self.mag1 = self.NO_MAG



    def dc_solution(self, magnitude_array, strike_array, dip_array, rake_array):
        lune_lat_rad = np.deg2rad(90)
        lune_long_rad = np.deg2rad(0)

        strike_array_rad = np.deg2rad(strike_array)
        dip_array_rad = np.deg2rad(dip_array)
        rake_array_rad = np.deg2rad(rake_array)

        for magnitude in magnitude_array:
            m0 = mag2moment(magnitude)
            for i, strike in enumerate(strike_array_rad):
                for j, dip in enumerate(dip_array_rad):
                    for k, rake in enumerate(rake_array_rad):
                        mt = m0 * DMT_enz(strike, dip, rake, lune_lat_rad, lune_long_rad)
                        _misfit = self.misfit(mt)

                        if self.min_misfit > _misfit:
                            self.strike = strike_array[i]
                            self.dip = dip_array[j]
                            self.rake = rake_array[k]
                            self.magnitude = magnitude
                            self.min_misfit = _misfit



    def mt_solution(self, lune_lat_array, lune_long_array):
        m0 = mag2moment(self.magnitude)

        self.lune_misfit_surface = np.zeros([len(lune_lat_array), len(lune_long_array)])
        lune_lat_array_rad = np.deg2rad(lune_lat_array)
        lune_long_array_rad = np.deg2rad(lune_long_array)

        for i, lune_lat in enumerate(lune_lat_array_rad):
            for j, lune_long in enumerate(lune_long_array_rad):
                mt = m0 * DMT_enz(np.deg2rad(self.strike), np.deg2rad(self.dip), np.deg2rad(self.rake),
                                  lune_lat, lune_long)
                _misfit = self.misfit(mt)
                self.lune_misfit_surface[i][j] = _misfit

                if self.min_misfit >= _misfit:
                    self.lune_lat = lune_lat_array[i]
                    self.lune_long = lune_long_array[j]
                    self.min_misfit = _misfit
