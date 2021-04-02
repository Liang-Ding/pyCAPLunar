# -------------------------------------------------------------------
# Full moment tensor
#
# refs
# Tape, W., & Tape, C. (2015). A uniform parametrization of moment tensors.
# GJI, 202(3), 2074–2081. https://doi.org/10.1093/gji/ggv262
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from MTTools.DLune import DCreate_Triple
from MTTools.DOrientations import DRotationMatrix, DRotz
import numpy as np

gRMax = DRotz(np.deg2rad(-45))
ginvRMax = np.linalg.inv(gRMax)

def DMomentTensor(strike, dip, rake, colatitude, longitude):
    '''
    Calculate the FULL moment tensor.

    :param strike:  The strike in radians, k in Carl's paper.  ranging [0, 2pi]
    :param dip:     The dip in radians, thetha in Carl's paper. ranging [0, pi/2]
    :param rake:    The slip (rake) in radians, sigma in Carl's paper. ranging [-pi/2, pi/2]
    :param colatitude: The colatitude in the Lune in radians (Beta), [0, PI]
    :param longitude:  The longitude in the Lune in radians (Gamma), [-PI/6, PI/6]
    :return: The full moment tensor.
    '''

    triple = DCreate_Triple(colatitude, longitude)
    triple_matrix = np.zeros((3, 3))
    triple_matrix[0, 0] = triple[0]
    triple_matrix[1, 1] = triple[1]
    triple_matrix[2, 2] = triple[2] 

    U = DRotationMatrix(strike, dip, rake)
    return np.dot(np.dot(U, triple_matrix), np.linalg.inv(U))


def DMT_enz(strike, dip, rake, colatitude, longitude):
    '''Calculate the moment tensor in e-n-z coord.'''
    mt_nwz = DMomentTensor(strike, dip, rake, colatitude, longitude)
    mt_enz = np.ones(6)
    mt_enz[0] = mt_nwz[1][1]
    mt_enz[1] = mt_nwz[0][0]
    mt_enz[2] = mt_nwz[2][2]

    # doubling is only for the rapid computation of synthetic waveform.
    mt_enz[3] = -1.0 * mt_nwz[0][1] * 2.0
    mt_enz[4] = -1.0 * mt_nwz[1][2] * 2.0
    mt_enz[5] = mt_nwz[0][2] * 2.0

    # scalar moment tensor to general seismic moment.
    # Peter M. Shearer, 2009, pp.247 equ.(9.8)
    return np.sqrt(2) * mt_enz

