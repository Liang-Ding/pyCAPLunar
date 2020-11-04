# -------------------------------------------------------------------
# Tools for running DCAPs package.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import numpy as np


def momentMagnitude_2_moment(magnitude):
    '''
    * Return the moment according the the magnitude.
    * ref: Formula (9.73), P. 284
    * Shearer, P. M. (2009). Introduction to Seismology,2nd Ed.
    * Cambridge: Cambridge University Press.
    '''
    return np.power(10, 1.5 * magnitude + 9.1)



def taper(data, taper_rate=0.1):
    '''Return the 1D tapered data.'''

    n_data = len(data)
    n_taper = int(n_data * taper_rate)

    # directly set the amplitude at the begnin and the end of data to zero. 
    taper_factors = np.hstack([np.zeros(n_taper), np.ones(n_data-2*n_taper), np.zeros(n_taper)])
    return taper_factors * data

    # taper by bell-shape function - will change the waveform, should not be used in the inversion. 
    # taper_amp_arr = np.asarray(0.5 * (1.0 - np.cos(np.arange(n_taper)*(np.pi/n_taper))))
    # return np.hstack([data[:n_taper]*taper_amp_arr, data[n_taper:-1*n_taper], data[-1*n_taper:]*np.flip(taper_amp_arr, axis=0)])



