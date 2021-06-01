# -------------------------------------------------------------------
# Cut phase segments from SGT data.
# Paste the synthetic waveform on the data. Cut the SGT only.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import numpy as np

# TODO: create a lite version of DCut.
def DCut(data, data_length, index0, index1):
    '''
    Return the cut segment data[index0:index1]

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


def DCut_sgt(sgt, n_t, n_length, n_offset):
    '''
    * Cut the Strain Green's Tensor according to the arrival time of phase.

    :param sgt:          The strain green's tensor of one stations.
                         Data Size=[n_step, n_dim, n_para]
    :param n_t:          The computational arrival time of phase in sample. INT.
    :param n_length:     The length of cut phase. INT.
    :param n_offset:     The offset before the arrival time. INT.
    :return:    The sliced SGT of phases: [phase_N_step, ndim, npara]
    '''

    n_step, n_dim, n_paras = np.shape(sgt)
    idx0 = int(n_t - n_offset)
    idx1 = int(idx0 + n_length)
    return DCut(data=sgt, data_length=n_step, index0=idx0, index1=idx1)


