# -------------------------------------------------------------------
# Function to calculate the waveform misfit.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import numpy as np

def DMisfit(syn, data):
    ''' Compute the waveform misfit. '''

    # Misfit function:
    # Zhu, L., & Helmberger, D. V. (1996). Advancement in source estimation. BSSA, 86(5), 1634–1641
    return np.sum(np.square(np.subtract(syn, data))) / np.dot(syn, data)


