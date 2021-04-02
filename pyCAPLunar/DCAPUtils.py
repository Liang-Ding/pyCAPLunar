# -------------------------------------------------------------------
# Tools for running pyCAPLunar package.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import numpy as np


def mag2moment(magnitude):
    '''
    * Return the moment according the the magnitude.
    * ref: Formula (9.73), P. 284
    * Shearer, P. M. (2009). Introduction to Seismology,2nd Ed.
    * Cambridge: Cambridge University Press.
    *
    * 1 newton meter (N-m)
    '''

    return np.power(10, 1.5 * magnitude + 9.1)
