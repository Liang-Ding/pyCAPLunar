# -------------------------------------------------------------------
# Calculate the orientation matrix for the given focal mechanism.
#
# refs
# Tape, W., & Tape, C. (2015). A uniform parametrization of moment tensors.
# Geophysical Journal International, 202(3), 2074–2081. https://doi.org/10.1093/gji/ggv262
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import numpy as np

def DRotx(angle):
    ''' Get rotation matrix at the x axis. angle in Radians. '''
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -1 * np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])

def DRoty(angle):
    ''' Get rotation matrix at the y axis. angle in Radians. '''
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-1 * np.sin(angle), 0, np.cos(angle)]])


def DRotz(angle):
    ''' Get rotation matrix at the z axis. angle in Radians. '''
    return np.array([[np.cos(angle), -1 * np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])


def DRotationMatrix(strike, dip, rake):
    '''
    * Calculate the rotation matrix for the given focal mechanism.
    * Using the Equ (10) in Carl's Paper (2015).

    :param strike: The strike in radians. k[kappa] ranging [0, 2pi]
    :param dip:     The dip in radians. the theta ranging [0, pi/2]
    :param rake:    The slip angle or the rake angle. the sigma ranging [-pi/2, pi/2]
    :return: the rotation matrix U(k(=strike), sigma(=rake), theta(=dip))
    '''
    Z_k = DRotz(-1.0 * strike)
    X_theta = DRotx(dip)
    Z_delta = DRotz(rake)
    tmp = np.dot(Z_k, X_theta)
    V = np.dot(tmp, Z_delta)

    Y_pi4 = DRoty(np.pi/-4.0)
    return np.dot(V, Y_pi4).round(6)

