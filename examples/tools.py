# -------------------------------------------------------------------
# Tools for the inversion.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import pandas as pd
import numpy as np


def find_minimum(file_path, b_segment=True):
    '''
    Return the iterm with minimum misift (1.0/misfit * data segment.)
    The first five data must be the following format:
    strike, dip, rake, magnitude, misfit, number of good segment, *, *, *, ...
    '''

    df = pd.read_csv(file_path, header=None)
    n_col = len(df.values[0, :])
    strike_arr = df.values[:, 0]
    dip_arr = df.values[:, 1]
    rake_arr = df.values[:, 2]
    mag_arr = df.values[:, 3]
    misfit_arr = df.values[:, 4]
    data_segment = df.values[:, 5]

    if b_segment:
        # consider the waveform misift and the number of good segment.
        idx = np.argmax(1.0 / misfit_arr * data_segment)
    else:
        # ONLY consider the waveform misfit!
        idx = np.argmin(misfit_arr)

    return strike_arr[idx], dip_arr[idx], rake_arr[idx], mag_arr[idx], \
               np.round(misfit_arr[idx], 7), int(data_segment[idx])


