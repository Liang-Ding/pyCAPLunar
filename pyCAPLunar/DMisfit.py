# -------------------------------------------------------------------
# Function to calculate the waveform misfit.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import numpy as np

def DMisfit(syn, data, type='Zhu1996'):
    ''' Compute the waveform misfit. '''

    # L2 Misfit
    if str(type).upper() == ('L2'):
        return np.average(np.square(np.subtract(syn, data)))

    # Misfit function:
    # Zhu, L., & Helmberger, D. V. (1996). Advancement in source estimation. BSSA, 86(5), 1634–1641
    else:
        return np.sum(np.square(np.subtract(syn, data))) / np.dot(syn, data)



