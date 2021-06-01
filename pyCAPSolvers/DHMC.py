# -------------------------------------------------------------------
# Uncertainty estimation by the Hamiltonian Monte Carlo method.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from MTTools.DMomentTensors import DMT_enz
from pyCAPLunar.DCAPUtils import mag2moment
from pyCAPLunar.DPrepare import _prepare_N_staiton_sgt, _prepare_N_staiton_data


from pyCAPLunar.DCAP import DCAP
import numpy as np
import pickle


class DHMC_DC(DCAP):
    '''DC component solver that is Based on the Hamiltonian Monte Carlo method. '''

    def __init__(self, sgts_list, data_list,
                 loc_src, loc_stations,
                 n_tp_sgt, n_ts_sgt,
                 n_tp_data, n_ts_data,
                 n_tp_max_shift, n_ts_max_shift,
                 dt, n_phase=5, n_component=3,
                 w_pnl=1.0, w_srf=1.0,
                 misfit_threshold=5.0,
                 reject_rate=1.0,
                 taper_scale=-0.4, taper_rate_srf=0.0,
                 cc_threshold=0.2,
                 amplitude_ratio_threshold=2.0,
                 job_id=None):

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
                 cc_threshold,
                 amplitude_ratio_threshold)


        # constant encoding the observational uncertainties ???
        self.sigma_d = 0.1
        self.sigma_q = np.inf

        # Double-couple component.
        # strike, dip, rake, magnitude.
        self.Nq = 4
        # delta to calculate derivatives.
        self.delta_q0 = 0.1 * np.ones(self.Nq)
        self.delta_q0[3] = 0.01  # magnitude

        self.b_initial_q0 = False
        self.MAX_SAMPLE = 100000

        # save samples
        self.b_save_samples = True
        self.saving_dir = None
        # save cache?
        self.b_save_cache = True
        self.n_step_cache = 10
        self.job_id = job_id



    def create_p(self):
        '''Using probability density function to create P. '''
        return np.random.normal(loc=0, scale=1.0, size=self.Nq)


    def Uq(self, q):
        '''Compute Uq. '''
        mt0 = mag2moment(q[3]) * DMT_enz(np.deg2rad(q[0]), np.deg2rad(q[1]),
                                             np.deg2rad(q[2]), np.deg2rad(90),
                                             np.deg2rad(0))

        misfit_matrix, _, _ = self.cut_and_paste(mt0)
        Uq = np.average(misfit_matrix[np.where(misfit_matrix != self.MAX_MISFIT)]) / (2.0 * np.power(self.sigma_d, 2))
        return Uq


    def Kp(self, p):
        '''Compute Kp, as M is identical matrix. '''
        return np.dot(p, p)/2.0

    def dUdq(self, q):
        '''Compute du_dqi'''
        self.dU_dqi=np.zeros(self.Nq)
        Uq = self.Uq(q)

        # dU/dqi
        for i in range(self.Nq):
            _q = q.copy()
            _q[i] += self.delta_q0[i]
            self.dU_dqi[i] = (self.Uq(_q) - Uq)/(self.delta_q0[i])

        # gradient for magnitude
        self.dU_dqi[3] = self.dU_dqi[3]/1000


    def set_q(self, q):
        '''
        Set initial solution.
        q0 = [strike, dip, rake, magnitude]
        '''

        if len(q) < self.Nq:
            print("Bad q0 with {} parameters, while {} are required. ".format(len(q), self.Nq))
            return

        self.q0 = np.zeros(self.Nq)  # 4 = cont([strike, dip, rake, mag])
        self.q0[0] = q[0]
        self.q0[1] = q[1]
        self.q0[2] = q[2]
        self.q0[3] = q[3]

        if not self.b_initial_q0:
            self.raw_q0 = self.q0.copy()

        self.b_initial_q0 = True
        self.dUdq(self.q0)


    def hmc(self, epsilon, n_step):
        if not self.b_initial_q0:
            print("Un-initial Q0!")
            return

        q = self.q0.copy()
        current_q = self.q0.copy()

        # initial p(t)
        p = self.create_p()
        current_p = p.copy()

        for i in range(n_step):
            # p(t+epsilon/2)
            p = p - (epsilon / 2.0) * self.dU_dqi
            # q(t+epsilon)
            q = q + epsilon * p
            self.set_q(q)
            p = p - (epsilon/2.0) * self.dU_dqi
        current_Uq = self.Uq(current_q)
        current_Kp = self.Kp(current_p)
        proposed_Uq = self.Uq(q)
        proposed_Kp = self.Kp(p)

        if 1.0 < np.exp(current_Uq-proposed_Uq+current_Kp-proposed_Kp):
            print("! Accept q={}".format(np.round(q, 2)))
            self.set_q(q)
            return q, True
        else:
            print("Reject, q={}".format(np.round(q, 2)))
            self.set_q(self.raw_q0.copy())
            return current_q, False


    def sampling(self, epsilon, n_step, n_sample):
        '''Sampling'''
        if not self.b_initial_q0:
            print("Un-initial Q0!")
            return

        if self.b_save_samples:
            # todo: checking directory
            if self.saving_dir is None:
                print("No saving directory available!")
                return

        samples = []
        _tmp_n_sample = 0
        while(len(samples) < n_sample and _tmp_n_sample < self.MAX_SAMPLE):
            _tmp_n_sample += 1

            # save the cache of sampling results.
            if self.b_save_cache:
                if len(samples)!=0 and np.mod(len(samples), self.n_step_cache) == 0:
                    self.save_to_file(name_str="cache_samples_N{}".format(len(samples)), samples=samples)
                    print("Cache including {} samples saved!".format(len(samples)))
            try:
                q, accept = self.hmc(epsilon, n_step)
                if accept:
                    samples.append(q)
            except:
                # reset
                self.set_q(self.raw_q0.copy())

        # write samples to file.
        if self.b_save_samples:
            self.save_to_file(name_str="Samples_N{}_".format(len(samples)), samples=samples)
        return samples


    def set_saving_dir(self, saving_dir):
        self.saving_dir = saving_dir

    def save_to_file(self, name_str, samples, format='.pkl'):
        '''Write out the sample to pkl file. '''
        if self.job_id is None:
            file_path = str(self.saving_dir) + str(name_str) + str(format)
        else:
            file_path = str(self.saving_dir) + str(self.job_id) + str('_') + str(name_str) + str(format)

        with open(file_path, 'wb') as f:
            pickle.dump(samples, f)



class DHMC_MT(DHMC_DC):
    '''Solver using Hamiltonian Monte Carlo. '''

    def __init__(self, sgts_list, data_list,
                 loc_src, loc_stations,
                 n_tp_sgt, n_ts_sgt,
                 n_tp_data, n_ts_data,
                 n_tp_max_shift, n_ts_max_shift,
                 dt, n_phase=5, n_component=3,
                 w_pnl=1.0, w_srf=1.0,
                 misfit_threshold=5.0,
                 reject_rate=1.0,
                 taper_scale=-0.4, taper_rate_srf=0.0,
                 cc_threshold=0.2,
                 amplitude_ratio_threshold=2.0,
                 job_id=None):

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
                 cc_threshold,
                 amplitude_ratio_threshold,
                         job_id)

        self.Nq = 6
        # delta to calculate derivatives.
        self.delta_q0 = 0.1 * np.ones(self.Nq)
        self.delta_q0[3] = 0.01  # magnitude


    def Uq(self, q):
        '''Compute Uq. '''
        mt0 = mag2moment(q[3]) * DMT_enz(np.deg2rad(q[0]), np.deg2rad(q[1]),
                                         np.deg2rad(q[2]), np.deg2rad(q[4]),
                                         np.deg2rad(q[5]))
        misfit_matrix, _, _ = self.cut_and_paste(mt0)
        Uq = np.average(misfit_matrix[np.where(misfit_matrix != self.MAX_MISFIT)]) / (2.0 * np.power(self.sigma_d, 2))
        return Uq


    def dUdq(self, q):
        '''Compute du_dqi'''
        self.dU_dqi=np.zeros(self.Nq)
        Uq = self.Uq(q)

        # dU/dqi
        for i in range(self.Nq):
            _q = q.copy()
            _q[i] += self.delta_q0[i]
            self.dU_dqi[i] = (self.Uq(_q) - Uq)/(self.delta_q0[i])

        # gradient for magnitude
        self.dU_dqi[3] = self.dU_dqi[3]/1000


    def set_q(self, q):
        ''' Set q=[strike, dip, rake, mag, lune_lat, lune_long]. '''
        self.q0 = q

        if not self.b_initial_q0:
            self.raw_q0 = self.q0.copy()

        self.b_initial_q0 = True
        self.dUdq(self.q0)



# moment tensor and location.
class DHMC_PRO(DHMC_MT):
    '''Solving the moment tensor and source location by the Hamiltonian Monte Carlo method. '''

    def __init__(self, sgtMgr, station_names,
                 sampling_rate,
                 p_freqmin, p_freqmax,
                 s_freqmin, s_freqmax,
                 n_p_length, n_p_offset,
                 n_s_length, n_s_offset,
                 vp, vs,
                 data_n_stations, loc_src, loc_stations,
                 n_tp_data,
                 n_tp_max_shift, n_ts_max_shift,
                 dt, n_phase=5, n_component=3,
                 w_pnl=1.0, w_srf=1.0,
                 misfit_threshold=5.0,
                 reject_rate=1.0,
                 taper_scale=-0.4, taper_rate_srf=0.0,
                 cc_threshold=0.2,
                 amplitude_ratio_threshold=2.0,
                 job_id=None):

        self.sgtMgr = sgtMgr
        self.station_names = station_names
        self.loc_stations = loc_stations
        self.sampling_rate = sampling_rate
        self.p_freqmin = p_freqmin
        self.p_freqmax = p_freqmax
        self.s_freqmin = s_freqmin
        self.s_freqmax = s_freqmax
        self. n_p_length = n_p_length
        self.n_p_offset = n_p_offset
        self.n_tp_data = n_tp_data
        self.n_s_length = n_s_length
        self.n_s_offset = n_s_offset
        self.vp = vp
        self.vs = vs

        try:
            sgts_list = self.get_sgt_list_n_stations(loc_src)
            data_list = self.get_data_list_n_stations(data_n_stations)
        except:
            print('Unable to initial the SGT and data.')
            return

        super().__init__(sgts_list, data_list,
                         loc_src, loc_stations,
                         self.n_tp_sgt, self.n_ts_sgt,
                         self.n_tp_data, self.n_ts_data,
                         n_tp_max_shift, n_ts_max_shift,
                         dt, n_phase, n_component,
                         w_pnl, w_srf,
                         misfit_threshold,
                         reject_rate,
                         taper_scale, taper_rate_srf,
                         cc_threshold,
                         amplitude_ratio_threshold,
                         job_id)


        # constant encoding the observational uncertainties ???
        self.sigma_d = 0.3
        self.sigma_q = np.inf

        # nine parameters [strike, dip, rake, magnitude, lune_latitude, lune_long, lat, long, depth]
        self.Nq = 9

        # delta to calculate derivatives.
        self.delta_q0 = 0.1 * np.ones(self.Nq)
        self.delta_q0[3] = 0.01  # magnitude

        # increment for latitude: q0[6], longitude: q0[7], depth: q0[8].
        # as the spacing of the mesh is 500 m.
        self.delta_q0[6] = 0.0051
        self.delta_q0[7] = 0.0051
        self.delta_q0[8] = 1.0

        self.b_initial_q0 = False



    def get_sgt_list_n_stations(self, loc_src):
        df = self.sampling_rate

        # loc_src=[lat, long, depth in km (the depth is minus)]
        source = loc_src.copy()
        source[2] = -1000 * np.fabs(source[2])

        lat, long, z, \
        utm_x, utm_y, utm_z, \
        idx_processor, element_index, \
        xi, eta, gamma = self.sgtMgr.find(x=source[0], y=source[1], z=source[2], n_points=1)

        # element index in the ibool file starts from 1, while 0 in our code.
        idx_element = element_index - 1

        try:
            self.sgtMgr.Initialize(idx_processor, idx_element, self.station_names)
            interp_sgt_n_stations = self.sgtMgr.get_sgt(source, mode='LAGRANGE')
        except:
            print("Unable to get SGT")
            return None


        # computing the approximate p and s travel time for cutting the SGT (synthetic waveform).
        self.n_ts_data = np.zeros_like(self.n_tp_data, dtype=int)
        n_tp_sgt_stations = []
        n_ts_sgt_stations = []
        for i, loc_sta in enumerate(self.loc_stations):
            dist = 111.0 * np.sqrt(np.square(loc_sta[0] - loc_src[0]) + np.square(loc_sta[1] - loc_src[1]))
            dist = np.sqrt(np.power(dist, 2) + np.power(loc_src[2] - loc_sta[2], 2))
            n_tp = int((dist / self.vp) * self.sampling_rate)
            n_ts = int((dist / self.vs) * self.sampling_rate)
            n_tp_sgt_stations.append(n_tp)
            n_ts_sgt_stations.append(n_ts)
            self.n_ts_data[i] = int(self.n_tp_data[i]) + (n_ts - n_tp)

        self.n_tp_sgt = self.n_p_offset * np.ones_like(n_tp_sgt_stations, dtype=int)
        self.n_ts_sgt = self.n_s_offset * np.ones_like(n_ts_sgt_stations, dtype=int)

        sgt_list_n_stations = _prepare_N_staiton_sgt(interp_sgt_n_stations, df,
                                                     self.p_freqmin, self.p_freqmax,
                                                     self.s_freqmin, self.s_freqmax,
                                                     n_tp_sgt_stations, self.n_p_length, self.n_p_offset,
                                                     n_ts_sgt_stations, self.n_s_length, self.n_s_offset)
        return sgt_list_n_stations


    def get_data_list_n_stations(self, data_n_stations):
        df = self.sampling_rate
        data_list_n_stations = _prepare_N_staiton_data(data_n_stations, df,
                                self.p_freqmin, self.p_freqmax,
                                self.s_freqmin, self.s_freqmax,
                                self.n_p_length, self.n_s_length,
                                self.n_tp_data, self.n_ts_data)
        return data_list_n_stations



    def Uq(self, q, b_sgt=False):
        '''Compute Uq. '''

        if b_sgt:
            sgt_list = self.get_sgt_list_n_stations(q[-3:])
            self.update_sgt(sgt_list)
            self.update_azimuth(q[-3:])

        mt0 = mag2moment(q[3]) * DMT_enz(np.deg2rad(q[0]), np.deg2rad(q[1]),
                                         np.deg2rad(q[2]), np.deg2rad(q[4]),
                                         np.deg2rad(q[5]))

        misfit_matrix, _, _ = self.cut_and_paste(mt0)
        Uq = np.average(misfit_matrix[np.where(misfit_matrix != self.MAX_MISFIT)]) / (2.0 * np.power(self.sigma_d, 2))
        return Uq


    def dUdq(self, q):
        '''Compute du_dqi'''
        self.dU_dqi=np.zeros(self.Nq)
        Uq = self.Uq(q)

        # original code to claculate dU/dqi
        # dU/dqi
        for i in range(self.Nq):
            _q = q.copy()
            _q[i] += self.delta_q0[i]
            if i >= 6:
                b_sgt = True
            else:
                b_sgt = False
            self.dU_dqi[i] = (self.Uq(_q, b_sgt) - Uq)/(self.delta_q0[i])


    def set_q(self, q):
        '''Set q. '''
        self.q0 = q
        if not self.b_initial_q0:
            self.raw_q0 = self.q0.copy()

        self.b_initial_q0 = True
        self.dUdq(self.q0)

