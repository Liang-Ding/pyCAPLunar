# -------------------------------------------------------------------
# Utils
# Extend the waveform. 
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import numpy as np

def DExtend_sub(data, lag):
    '''
    Return the extended data.
    Data should be separated by the first index.

    :param data: The data to be extended.
    :param lag:  The number of lag.
    :return:    The extended data.
    '''

    if lag == 0:
        return data

    n_lag = int(np.fabs(lag))

    idx = np.hstack([n_lag, np.shape(data)[1:]])
    if len(idx) == 1:
        data_zeros = np.zeros(int(idx[0]))
        if lag < 0:
            data_ext = np.hstack([data, data_zeros])
        else:
            data_ext = np.hstack([data_zeros, data])
    else:
        data_zeros = np.zeros(idx)
        if lag < 0:
            data_ext = np.vstack([data, data_zeros])
        else:
            data_ext = np.vstack([data_zeros, data])

    return data_ext



