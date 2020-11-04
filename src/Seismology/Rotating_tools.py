# -------------------------------------------------------------------
# Waveform rotation tools.
#
# The function rotate_ne_rt() is revised from
# the rotate_ne_rt() in obspy package.
# (ref: obspy.signal.rotate.rotate_ne_rt())
#
# :copyright:
#     The ObsPy Development Team (devs@obspy.org)
# :license:
#     GNU Lesser General Public License, Version 3
#     (https://www.gnu.org/copyleft/lesser.html)
# -------------------------------------------------------------------


import numpy as np


def rotate_ne_rt(n, e, ba):
    """
    Rotates horizontal components of a seismogram.

    The North- and East-Component of a seismogram will be rotated in Radial
    and Transversal Component. The angle is given as the back-azimuth, that is
    defined as the angle measured between the vector pointing from the station
    to the source and the vector pointing from the station to the North.

    :type n: :class:`~numpy.ndarray`
    :param n: Data of the North component of the seismogram.
    :type e: :class:`~numpy.ndarray`
    :param e: Data of the East component of the seismogram.
    :type ba: float
    :param ba: The back azimuth from station to source in degrees.
    :return: Radial and Transversal component of seismogram.
    """
    if len(n) != len(e):
        raise TypeError("North and East component have different length.")
    if ba < 0 or ba > 360:
        raise ValueError("Back Azimuth should be between 0 and 360 degrees.")
    ba = np.deg2rad(ba)
    r = - e * np.sin(ba) - n * np.cos(ba)
    t = - e * np.cos(ba) + n * np.sin(ba)
    return r, t



def DRotate_XY2RT(x_arr, y_arr, ba):
    '''
    Rotate the waveform from X-Y (E-N) to R-T dirction
    :param x_arr: The wavefrom on X(E) direction.
    :param y_arr: The wavefrom on Y(N) direction.
    :param ba:  back azimuth from station to source in degrees.
    :return: rotate_ne_rt
    '''
    return rotate_ne_rt(y_arr, x_arr, ba)


def DRotate_EN2RT(e_arr, n_arr, ba):
    '''Rotate the waveform from X-Y (E-N) to R-T dirction.'''
    return rotate_ne_rt(n_arr, e_arr, ba)


def DRotate_XYZ2RTZ(xyz_arr, ba):
    '''
    * Rotate the XYZ array to RTZ.

    :param xyz_arr: 3C array in X-Y-Z (E-N-Z), size=[3, n_sample]
    :return: 3C array in R-T-Z.
    '''
    rtz_arr = np.zeros_like(xyz_arr)
    rtz_arr[0], rtz_arr[1] = DRotate_XY2RT(xyz_arr[0], xyz_arr[1], ba)
    rtz_arr[2] = xyz_arr[2]
    return rtz_arr


def DRotate_ENZ2RTZ(enz_arr, ba):
    '''
    * Rotate the XYZ array to RTZ.
    '''
    return DRotate_XYZ2RTZ(enz_arr, ba)
