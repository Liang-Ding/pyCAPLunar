# -------------------------------------------------------------------
# Utils
# Tools to operate focal mechanism. 
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import numpy as np


def DFocalPlane_2_mt(strike, dip, rake, m0=1.0):
    '''
    * Return the moment tensor according to the strike-dip-rake.
    * ref: Jost, M. L., & Herrmann, R. B. (1989). A Student’s Guide to and Review of Moment Tensors.
     Seismological Research Letters, 60(2), 37–57. https://doi.org/10.1785/gssrl.60.2.37
    * from N-E-Z to E-N-Z: x = y, y = x, z = -z

    :param strike: The strike in radian (2 \pi rad equals 360 degrees).
    :param dip:    The dip in radian (2 \pi rad equals 360 degrees).
    :param rake:   The rake in radian (2 \pi rad equals 360 degrees).
    :param m0:     The moment magnitude.
    :return: The moment tensor in E-N-Z system.
    '''

    mt = np.zeros(6)
    # Mxx
    mt[0] = m0 * (np.sin(dip)*np.cos(rake)*np.sin(2.0*strike)
                  - np.sin(2.0*dip)*np.sin(rake)*np.square(np.cos(strike)))

    # Myy
    mt[1] = -1.0 * m0 * (np.sin(dip) * np.cos(rake) * np.sin(2.0*strike)
                  + np.sin(2.0*dip)*np.sin(rake)*np.square(np.sin(strike)))

    # Mzz
    mt[2] = m0 * np.sin(2.0 * dip) * np.sin(rake)

    # Mxy
    mt[3] = m0 * (np.sin(dip)*np.cos(rake)*np.cos(2.0*strike)
                  + 0.5*np.sin(2.0*dip)*np.sin(rake)*np.sin(2.0*strike))

    # Mxz
    mt[4] = m0 * (np.cos(dip)*np.cos(rake)*np.sin(strike)
                  - np.cos(2.0*dip)*np.sin(rake)*np.cos(strike))

    # Myz
    mt[5] = m0 * (np.cos(dip)*np.cos(rake)*np.cos(strike)
                  + np.cos(2.0*dip)*np.sin(rake)*np.sin(strike))


    return mt
    
