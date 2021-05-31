# -------------------------------------------------------------------
# ibool reader.
# *x.bin, *y.bin, *z.bin
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from DSEM_Utils import NGLLX, NGLLY, NGLLZ, CONSTANT_INDEX_27_GLL
from scipy.io import FortranFile
import numpy as np


def read_ibool_by_scipy(ibool_file, NSPEC_global):
    '''
    * Return the index data (will be ordered from 0 by FORCE) in the .ibool file.

    :param file_path:    The path to the .ibool file.
    :param NSPEC_global: The number of element stored in the .ibool file.
    :return:
            The data array in the .ibool file.
    '''

    f = FortranFile(ibool_file, 'r')
    ibool = f.read_reals(dtype='int32')
    f.close()
    ibool = np.reshape(ibool, (NSPEC_global, NGLLX * NGLLY * NGLLZ))

    # ATTENTION
    # The index in .ibool files starts from 1.
    ibool = ibool - 1

    return ibool


def extract_element_125(ibool_file, NSPEC_global, index_element):
    '''
    * Extract the global index of all 125 GLL points in the selected element.

    :param file_path:       The file path of the ibool file. Data type: STR.
    :param NSPEC_global:    The number of element in total. Data type: int.
    :param index_element:   The index of the selected element.
    :return:
            The global index of the selected 125 GLL points in the selected element.
            Data type: numpy.ndarray.
    '''

    ibool = read_ibool_by_scipy(ibool_file, NSPEC_global)
    if ibool.__len__() <= index_element:
        return np.zeros(NGLLX*NGLLY*NGLLZ)

    else:
        return ibool[index_element]


def extract_element_27(ibool_file, NSPEC_global, index_element):
    '''
    * Extract the global index of the 27 GLL points in the selected element.

    :param file_path:       The file path of the ibool file. Data type: STR.
    :param NSPEC_global:    The number of element in total. Data type: int.
    :param index_element:   The index of the selected element.
    :return:
            The global index of the selected 27 GLL points in the selected element.
            Data type: numpy.ndarray.
    '''

    ibool = read_ibool_by_scipy(ibool_file, NSPEC_global)
    if ibool.__len__() <= index_element:
        return np.zeros(27)

    else:
        NGLLX_N3 = 3
        NGLLY_N3 = 3
        NGLLZ_N3 = 3

        # get the array of the global index of selected GLL points.
        gll_array = ibool[index_element][CONSTANT_INDEX_27_GLL]

        # sort the index of GLL points.
        gll_points = []
        gll_array = np.reshape(gll_array, [NGLLZ_N3, NGLLY_N3, NGLLX_N3])
        for i in range(NGLLX_N3):
            for j in range(NGLLY_N3):
                for k in range(NGLLZ_N3):
                    gll_points.append(gll_array[k, j, i])
        gll_points = np.asarray(gll_points)

        return gll_points



def DEnquire_Element(ibool_file, NSPEC_global, index_element, nGLL_per_element=-1):
    '''
    * Extract the global index of GLL points in the selected element.

    :param file_path:        The file path of the ibool file. string.
    :param NSPEC_global:     The number of element in the slice. int.
    :param index_element:    The index of the selected element. int.
    :param nGLL_per_element: The factor controlled the number of GLL point extracted from the element.
                             if -1, extract all 125 gll points.
    :return:
            The global index of the GLL points in the element.
            Data type: numpy.ndarray.
    '''

    if 27 == nGLL_per_element:
        return extract_element_27(ibool_file, NSPEC_global, index_element)
    else:
        return extract_element_125(ibool_file, NSPEC_global, index_element)
