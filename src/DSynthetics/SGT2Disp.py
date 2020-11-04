# -------------------------------------------------------------------
# Compute the displacement. 
# displacement = Moment tensor * SGT.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from Seismology.Rotating_tools import DRotate_ENZ2RTZ
import numpy as np


def SGTs_2_Displacement_1C(mt, SGT):
    '''
    * Return the 1-C Synthetic displacement of moment tensor (mt) and Strain Green functions(SGTs)

    * XYZ to ENZ
    * x -> E
    * y -> N
    * z -> Z

    # xyz to rtp
    r = z
    theta = -y
    phi = x

    * r-theta-phi to XYZ
    * Mrr = Mzz
    * Mtt = Myy
    * Mpp = Mxx
    * Mrt = -Myz
    * Mrp = Mxz
    * Mtp = -Mxy

    # xyz - ENZ - rtp
    # [xx, yy, zz, xy, xz, yz] = [EE, NN, ZZ, EN, EZ, NZ]
    # [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    # [Mpp, Mtt, Mrr, -Mtp, Mrp, -Mrt]

    :param mt: The moment tensor elements [Mxx, Myy, Mzz, Mxy, Mxz, Myz], 1-D array, length=6.
    :param SGTs:  The Strain green functions [[Exx_1,Eyy_1, Ezz_1, Exy_1, Exz_1, Eyz_1],
                                              [Exx_1,Eyy_1, Ezz_1, Exy_1, Exz_1, Eyz_1],
                                              ... ], 2-D array, shape: nx6.

    :return:
    '''
    # check data validation.
    length, n_comp = np.shape(SGT)
    if n_comp != 6:
        return False

    new_mt = mt + 0.0
    # double Mij, when i!=j.
    new_mt[3:6] = 2.0 * new_mt[3:6]

    return np.dot(SGT, new_mt)


def SGTs_2_Displacement(mt, SGTs):
    '''
    * Return the 3-C synthetic displacement array from MT and SGTs.

    :param mt:   The moment tensor elements.
                 * Size = [6, ] (Mxx, Myy, Mzz, Mxy, Mxz, Myz)
                 * 6: the number of item in moment tensor matrix.
                 * 1-D array

    :param SGTs: The Strain green functions of Unit Force X, Y, and Z.
                 * Size = [T, 3, 6],
                 * T: total time steps.
                 * 3: the number of unit force. In three orthogonal directions.
                 * 6: the number of element in epsilon matrix. [xx, yy, zz, xy, xz, yz]

    :return:    The 3-C synthetic displacements.
                * Size = [3, T]
                * 3: the number of component.
                * T: total time steps.
    '''

    displacement = []
    n_step, n_component, n_para = np.shape(SGTs)
    if n_component != 3:
        return None

    if n_para != 6:
        return None

    for idx in range(n_component):
        res = SGTs_2_Displacement_1C(mt, SGTs[:, idx, :])
        if res is not None:
            displacement.append(res)
        else:
            return None


    # !!! ATTENTION: the SGTs are stored as N-E-Z format in our study.
    # N-E-Z to E-N-Z
    u_nez_arr = np.vstack(displacement)
    u_enz = np.zeros_like(u_nez_arr)
    u_enz[1] = u_nez_arr[0]
    u_enz[0] = u_nez_arr[1]
    u_enz[2] = u_nez_arr[2]

    return u_enz



def SGTs_2_Displacement_RTZ(mt, SGTs, ba):
    '''
    * Synthetic 3C displacement in RTZ direction.

    :param mt:
    :param SGTs:
    :param ba: The back azimuth from station to source in degrees.
    :return:
    '''

    u_enz = SGTs_2_Displacement(mt, SGTs)
    return DRotate_ENZ2RTZ(u_enz, ba)



