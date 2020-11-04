# -------------------------------------------------------------------
# Align the waveform and SGTs by CC.
#
# Ref:
# Zhu, L., & Helmberger, D. V. (1996).
# Advancement in source estimation techniques using broadband regional seismograms.
# Bulletin of the Seismological Society of America, 86(5), 1634–1641.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from DCAPs.DCAPExtend import DExtend
from DCAPs.DCAPUtils import taper
from Seismology.DFocalTools import DFocalPlane_2_mt
from DSynthetics.SGT2Disp import SGTs_2_Displacement_RTZ, SGTs_2_Displacement

import numpy as np


def valid_length(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def DShift_1d(template, data, b_taper=False):
    '''
    * Return the Time lag between the template and the data.

    :param template: Template array. numpy 1D array.
    :param data:     Data, numpy 1D array.
    :param b_taper:
    :return:
    '''

    if len(template) != len(data):
        print("Unbalanced length")
        return None, None

    if b_taper:
        synthetic = taper(template)
        dat = taper(data)
    else:
        synthetic = template + 0.0
        dat = data + 0.0

    len_syn = len(synthetic)
    len_dat = len(dat)

    vlen_syn = valid_length(len_syn)
    vlen_dat = valid_length(len_dat)

    vlength = np.max([vlen_syn, vlen_dat])
    vlength_2 = 2*vlength

    new_template = np.hstack([synthetic, np.zeros(vlength_2 - len_syn)])
    new_data = np.hstack([np.zeros(vlength_2 - len_dat), dat])

    new_template_max = np.max(np.fabs(new_template))
    new_data_max = np.max(np.fabs(new_data))

    tmp_ft = np.fft.fft(new_template/new_template_max)
    data_ft = np.fft.fft(new_data/new_data_max)
    r = np.fft.ifft(np.multiply(np.conj(data_ft), tmp_ft))
    idx_max = np.argmax(r)

    lag = len_syn - idx_max

    if lag < 0:
        tmp1 = np.hstack([np.zeros(np.int(np.fabs(lag))), dat])
        tmp2 = np.hstack([synthetic, np.zeros(np.int(np.fabs(lag)))])
    else:
        tmp1 = np.hstack([dat, np.zeros(np.int(lag))])
        tmp2 = np.hstack([np.zeros(np.int(lag)), synthetic])

    tmp1_max = np.max(np.fabs(tmp1))
    tmp2_max = np.max(np.fabs(tmp2))
    cc = np.corrcoef(tmp1/tmp1_max, tmp2/tmp2_max)[0, 1]

    return lag, cc







