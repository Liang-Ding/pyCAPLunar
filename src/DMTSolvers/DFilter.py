# -------------------------------------------------------------------
# Bandpass filters.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from Seismology.DFilter import bandpass
import numpy as np


def DFilter_sgt(sgt_array, freqmin, freqmax, df):
    '''
    * Return filtered SGTs

    :param sgt_array:   The sgt array for one station.
                        Data shape: [n_sample, n_dim, n_para]
    :param freqmin:     The low freq limit of the bandpass filter.
    :param freqmax:     The high freq limit of the bandpass filter.
    :param df:          The sampling rate in Hz.
    :return:
    '''

    # constant
    n_dim = 3
    n_paras = 6
    new_sgt_array = np.zeros_like(sgt_array)
    for i in range(n_dim):
        for j in range(n_paras):
            new_sgt_array[:, i, j] = bandpass(sgt_array[:, i, j], freqmin=freqmin, freqmax=freqmax,
                                              df=df, corners=4, zerophase=False)
    return new_sgt_array




def DFilter_data(data_arr, freqmin, freqmax, df):
    '''
    * Return filtered data

    :param data_arr:   The data array from one station.
                        Data shape: [n_component, n_sample]
    :param freqmin:     The low freq limit of the bandpass filter.
    :param freqmax:     The high freq limit of the bandpass filter.
    :param df:          The sampling rate in Hz.
    :return:
    '''
    new_data_arr = np.zeros_like(data_arr)
    for i, tr in enumerate(data_arr):
        new_data_arr[i] = bandpass(tr, freqmin=freqmin, freqmax=freqmax,
                                   df=df, corners=4, zerophase=False)

    return new_data_arr
