# -------------------------------------------------------------------
# Filters.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from obspy.signal.filter import bandpass
import numpy as np


def DFilter_sgt(sgt, freqmin, freqmax, df):
    '''
    * Return the filtered SGTs

    :param sgt:         The sgt array for one station.
                        Data shape: [n_sample, n_dim, n_para]
    :param freqmin:     The low freq limit of the bandpass filter.
    :param freqmax:     The high freq limit of the bandpass filter.
    :param df:          The sampling rate in Hz.
    :return:
    '''

    # constant
    n_dim = 3
    n_paras = 6
    new_sgt = np.zeros_like(sgt)
    for i in range(n_dim):
        for j in range(n_paras):
            new_sgt[:, i, j] = bandpass(sgt[:, i, j], freqmin=freqmin, freqmax=freqmax,
                                              df=df, corners=4, zerophase=False)
    return new_sgt




def DFilter_data(data, freqmin, freqmax, df):
    '''
    * Return filtered data

    :param data:   The data array from one station.
                        Data shape: [n_component, n_sample]
    :param freqmin:     The low freq limit of the bandpass filter.
    :param freqmax:     The high freq limit of the bandpass filter.
    :param df:          The sampling rate in Hz.
    :return:
    '''
    new_data = np.zeros_like(data)
    for i, tr in enumerate(data):
        new_data[i] = bandpass(tr, freqmin=freqmin, freqmax=freqmax,
                                   df=df, corners=4, zerophase=False)

    return new_data