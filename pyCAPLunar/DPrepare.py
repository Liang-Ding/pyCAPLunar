# -------------------------------------------------------------------
# Prepare the data and SGT to conduct the inversion.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from pyCAPLunar import g_TAPER_SCALE, g_TAPER_RATE_SRF
from pyCAPLunar.DFilters import DFilter_sgt, DFilter_data
from pyCAPLunar.DCut import DCut_sgt
import numpy as np


def prepare_sgt(sgt, df, p_freqmin, p_freqmax,
                s_freqmin, s_freqmax,
                n_tp, n_p_length, n_p_offset,
                n_ts, n_s_length, n_s_offset):
    '''
    Filter and cut SGT.

    :param sgt:         The sgt array for one station.
                        Data shape: [n_sample, n_dim, n_para]
    :param p_freqmin:   The low freq limit of the bandpass filter for the PNL.
    :param p_freqmax:   The high freq limit of the bandpass filter for the PNL.
    :param s_freqmin:   The low freq limit of the bandpass filter for the S/Surface.
    :param s_freqmax:   The high freq limit of the bandpass filter for the S/Surface.
    :param df:          The sampling rate in Hz.
    :param n_tp:        The computational arrival time of the Pnl phase in sample. INT.
    :param n_p_length:  The length of the cut Pnl phase. INT.
    :param n_p_offset:  The offset in sample before the arrival time of the Pnl phase. INT.
    :param n_ts:        The computational arrival time of the S/Surface phase in sample. INT.
    :param n_s_length:  The length of the cut S/Surface phase. INT.
    :param n_s_offset:  The offset in sample before the arrival time of the S/Surface phase. INT.
    :return:            A list of the cut and filtered SGT data of phases.
                        Data shape: [[PNL_step, ndim, npara], [S/Surface_step, ndim, npara]
    '''

    sgt_list = []
    # pnl
    sgt_list.append(DCut_sgt(DFilter_sgt(sgt, p_freqmin, p_freqmax, df), n_tp, n_p_length, n_p_offset))
    # s/surface
    sgt_list.append(DCut_sgt(DFilter_sgt(sgt, s_freqmin, s_freqmax, df), n_ts, n_s_length, n_s_offset))

    return sgt_list


def prepare_data(data, df, p_freqmin, p_freqmax,
                s_freqmin, s_freqmax, n_tp, n_ts,
                 n_p_length, n_s_length, b_taper=True):
    '''
    Preapre the data for single station.
    Only Filter.

    :param data:        The 3C waveform for one station. Data size: [3 * n_sample]
    :param p_freqmin:   The low freq limit of the bandpass filter for the PNL.
    :param p_freqmax:   The high freq limit of the bandpass filter for the PNL.
    :param s_freqmin:   The low freq limit of the bandpass filter for the S/Surface.
    :param s_freqmax:   The high freq limit of the bandpass filter for the S/Surface.
    :param df:          The sampling rate in Hz.
    :param n_tp:        The arrival time of p phase.
    :param n_ts:        The arrival tiem of S phase.
    :return:            The data list for one staiton [[PNL: 3 * n_sample], [S/Surface: 3 * n_sample]]
    '''

    data_list = []

    if b_taper:
        # taper by clip
        _taper_p = DFilter_data(data, p_freqmin, p_freqmax, df)
        for i in range(len(_taper_p)):
            _taper_p[i][:n_tp] = _taper_p[i][:n_tp] * np.exp(g_TAPER_SCALE * np.arange(n_tp, 0, -1))
            _taper_p[i][n_tp+n_p_length:] = _taper_p[i][n_tp+n_p_length:] * np.exp(g_TAPER_SCALE * np.arange(len(_taper_p[i])-(n_tp+n_p_length)))

        _taper_s = DFilter_data(data, s_freqmin, s_freqmax, df)
        for i in range(len(_taper_s)):
            # find the peak of the S/Surface after the S arrival.
            _length = len(_taper_s[i])
            if n_ts+n_s_length < _length:
                ts_peak = np.argmax(np.fabs(_taper_s[i, n_ts:n_ts + n_s_length]))
            else:
                ts_peak = np.argmax(np.fabs(_taper_s[i, n_ts:n_ts + n_s_length]))
            # taper the surface waveform at the left of the peak.
            _taper_idx0 = n_ts + int(ts_peak * g_TAPER_RATE_SRF)
            _taper_s[i][:_taper_idx0] = _taper_s[i][:_taper_idx0] * np.exp(g_TAPER_SCALE * np.arange(_taper_idx0, 0, -1))
            # taper the surface waveform at the right of the peak.
            _taper_idx1 = n_ts + int((2.0 - g_TAPER_RATE_SRF) * ts_peak)
            if _taper_idx1 < _length:
                _taper_s[i][_taper_idx1:] = _taper_s[i][_taper_idx1:] * np.exp(g_TAPER_SCALE * np.arange(_length - _taper_idx1))

        data_list.append(_taper_p)
        data_list.append(_taper_s)
    else:
        # without taper.
        data_list.append(DFilter_data(data, p_freqmin, p_freqmax, df))
        data_list.append(DFilter_data(data, s_freqmin, s_freqmax, df))

    return data_list


def _prepare_N_staiton_sgt(sgt_n_stations, df,
                           p_freqmin, p_freqmax,
                           s_freqmin, s_freqmax,
                           n_tp_sgt_stations, n_p_length, n_p_offset,
                           n_ts_sgt_stations, n_s_length, n_s_offset):
    sgt_list_n_stations = []
    n_station = len(sgt_n_stations)
    for i in range(n_station):
        # sgt
        sgt_list_n_stations.append(prepare_sgt(sgt_n_stations[i], df, p_freqmin, p_freqmax,
                                               s_freqmin, s_freqmax,
                                               n_tp_sgt_stations[i], n_p_length, n_p_offset,
                                               n_ts_sgt_stations[i], n_s_length, n_s_offset))
    return sgt_list_n_stations


def _prepare_N_staiton_data(data_n_stations, df,
                            p_freqmin, p_freqmax,
                            s_freqmin, s_freqmax,
                            n_p_length, n_s_length,
                            n_tp_data, n_ts_data):
    data_list_n_stations = []
    n_station = len(data_n_stations)
    for i in range(n_station):
        # data
        data_list_n_stations.append(prepare_data(data_n_stations[i], df, p_freqmin, p_freqmax,
                                                 s_freqmin, s_freqmax, n_tp_data[i], n_ts_data[i],
                                                 n_p_length, n_s_length))
    return data_list_n_stations


def prepare_N_staiton(sgt_n_stations, data_n_stations, df,
                      p_freqmin, p_freqmax,
                      s_freqmin, s_freqmax,
                      n_tp_sgt_stations, n_p_length, n_p_offset,
                      n_ts_sgt_stations, n_s_length, n_s_offset,
                      n_tp_data, n_ts_data):
    '''
    Prepare the SGT and data at stations.

    :param sgt_n_staitons:    The sgt at N stations.
                             Data size: [[time_sample, n_dim, n_para] at station 1,
                                         [time_sample, n_dim, n_para] at station 2, ...]
    :param data_n_stations:   The observation data at N stations.
                             Data size: [[3 * n_sample] at staion 1,
                                         [3 * n_sample] at staion 2, ...]
    :param df:          The sampling rate in Hz.
    :param p_freqmin:   The low freq limit of the bandpass filter for the PNL.
    :param p_freqmax:   The high freq limit of the bandpass filter for the PNL.
    :param s_freqmin:   The low freq limit of the bandpass filter for the S/Surface.
    :param s_freqmax:   The high freq limit of the bandpass filter for the S/Surface.
    :param n_tp_sgt_stations:        The computational arrival time of the Pnl phase for SGT in sample at all stations. Array.
    :param n_p_length:  The length of the cut Pnl phase. INT.
    :param n_p_offset:  The offset in sample before the arrival time of the Pnl phase. INT.
    :param n_ts_sgt_stations:        The computational arrival time of the S/Surface phase for SGT in sample at all stations. Array.
    :param n_s_length:  The length of the cut S/Surface phase. INT.
    :param n_s_offset:  The offset in sample before the arrival time of the S/Surface phase. INT.
    :param n_tp_data:   The arrival time of p phase on the data
    :param n_ts_data:   The arrival time of S phase on the data.
    :return:
    sgt_list_n_stations:    The sgt of the PNL and s/surface at N stations.
                            Data size: [
                            [[time_sample, n_dim, n_para]_PNL, [time_sample, n_dim, n_para]_S*] at sta 1.],
                            [[time_sample, n_dim, n_para]_PNL, [time_sample, n_dim, n_para]_S*] at sta 2.],
                            ... ]
    data_list_n_statitons:  The data duplicated for fitting the PNL and S/Surface.
                            Data size: [  [[3* sample]_PNL, [3 * sample]_S] at station 1.
                                          [[3* sample]_PNL, [3 * sample]_S] at station 2.
                                        ...]
    '''

    # check data
    if len(sgt_n_stations) != len(data_n_stations):
        print("Incompatible number of station between the data and SGT. ")
        return False

    # initialize the data_container
    sgt_list_n_stations = _prepare_N_staiton_sgt(sgt_n_stations, df,
                      p_freqmin, p_freqmax,
                      s_freqmin, s_freqmax,
                      n_tp_sgt_stations, n_p_length, n_p_offset,
                      n_ts_sgt_stations, n_s_length, n_s_offset)


    data_list_n_stations = _prepare_N_staiton_data(data_n_stations, df,
                      p_freqmin, p_freqmax,
                      s_freqmin, s_freqmax,
                      n_p_length, n_s_length,
                      n_tp_data, n_ts_data)

    return sgt_list_n_stations, data_list_n_stations




