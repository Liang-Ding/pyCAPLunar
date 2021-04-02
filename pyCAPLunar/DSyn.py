# -------------------------------------------------------------------
# Synthetic and rotation.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import numpy as np

def DSynRotateRTZ(mt, SGTs, ba_rad, n_lens):
    '''

    :param mt:      The calibrated moment tensor: [Mxx, Myy, Mzz, 2*Mxy, 2*Mxz, 2*Mxz]
    :param SGTs:    The strain Green's tensor. ordered in N-E-Z
    :param ba_rad:  The back azimuth in radius.
    :n_lens:        The length of the SGT data.
    :return:
    '''

    u_rtz = np.zeros((3, n_lens))
    e = np.dot(SGTs[:, 1, :], mt)
    n = np.dot(SGTs[:, 0, :], mt)
    u_rtz[2] = np.dot(SGTs[:, 2, :], mt)
    u_rtz[0] = - e * np.sin(ba_rad) - n * np.cos(ba_rad)
    u_rtz[1] = - e * np.cos(ba_rad) + n * np.sin(ba_rad)
    return u_rtz


def DENZ2RTZ(enz_arr, ba_rad):
    '''
    Convert the data from E-N-Z to R-T-Z system.
    :param enz_arr: The data in E-N-Z
    :param ba_rad:  The back azimuth in radius.
    :return: Data in R-T-Z.
    '''

    rtz_arr = np.zeros_like(enz_arr)

    rtz_arr[2] = enz_arr[2]
    rtz_arr[0] = - enz_arr[0] * np.sin(ba_rad) - enz_arr[1] * np.cos(ba_rad)
    rtz_arr[1] = - enz_arr[0] * np.cos(ba_rad) + enz_arr[1] * np.sin(ba_rad)

    return rtz_arr




