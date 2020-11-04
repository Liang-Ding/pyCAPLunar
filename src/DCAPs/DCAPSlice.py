# -------------------------------------------------------------------
# Utils
# Slice the waveform into segments.
#
# Ref:
# Zhu, L., & Helmberger, D. V. (1996).
# Advancement in source estimation techniques using broadband regional seismograms.
# Bulletin of the Seismological Society of America, 86(5), 1634–1641.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import numpy as np


def DSlice_data(data, data_length, index0, index1):
    '''
    Return the sliced segments: data[index0:index1]

    :param data:        The data, separated by the first index.
    :param data_length: The length of the data in the first index.
    :param index0:      The first index of the slice.
    :param index1:      The last index of the slice.
    :return:
    '''

    sub_data = data.copy()
    b_extend_left = False
    if index0 < 0:
        n_ext = int(np.fabs(index0))
        idx = np.hstack([n_ext, np.shape(data)[1:]])
        tmp_arr = np.zeros(idx)
        tmp_data = np.vstack([tmp_arr, sub_data])
        sub_data = tmp_data
        idx0 = 0
        b_extend_left = True
    else:
        idx0 = int(index0)

    if index1 > data_length:
        n_ext = int(index1-data_length)
        idx = np.hstack([n_ext, np.shape(data)[1:]])
        tmp_arr = np.zeros(idx)
        tmp_data = np.vstack([sub_data, tmp_arr])
        sub_data = tmp_data
        idx1 = len(sub_data)
    else:
        if b_extend_left:
            idx1 = int(index1) - int(index0)
        else:
            idx1 = int(index1)

    return sub_data[int(idx0):int(idx1)]

  

