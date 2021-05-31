# -------------------------------------------------------------------
# XYZ reader.
# *x.bin, *y.bin, *z.bin
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from DSEM_Utils.DWidgets import get_proc_name
from DSEM_Utils import CONSTANT_INDEX_27_GLL
from DSEM_Utils.ibool_reader import read_ibool_by_scipy
from scipy.io import FortranFile


def read_xyz_bin_by_scipy(file_path):
    '''
    * read XYZ bin file by using the SciPy package.

    :param file_path: The path of the XYZ bin file.
    :param file_type: The data type of the XYZ bin.

    * x,y,z -- float32

    :return: The whole data.
    '''

    data_type = 'float32'

    try:
        f = FortranFile(file_path, 'r')
        dat = f.read_reals(dtype=data_type)
        f.close()
    except:
        print("Unable to open file: ", str(file_path))
        return None
    return dat


def DEnquire_XYZ_GLLs_Element(data_dir, idx_processor, idx_element,
                              NSPEC_global, nGLL_per_element=27):
    '''
    * return the x, y, z of the gll points in a specific specific element.

    :param data_dir:        The dir of the *.bin files.
    :param idx_proc:        The index of the processor. INT
    :param idx_element:     The index of the element in the processor. INT
    :param NSPEC_global:    The number of the element in the processor. INT
    :return:                The x, y, and z array of the GLL points.
    '''

    proc_name = get_proc_name(idx_processor)
    ibool_file = str(data_dir) + proc_name + "_ibool.bin"
    x_file = str(data_dir) + proc_name + "_x.bin"
    y_file = str(data_dir) + proc_name + "_y.bin"
    z_file = str(data_dir) + proc_name + "_z.bin"

    # read ibool file
    ibool = read_ibool_by_scipy(ibool_file, NSPEC_global)
    # read x, y, z
    x = read_xyz_bin_by_scipy(x_file)
    y = read_xyz_bin_by_scipy(y_file)
    z = read_xyz_bin_by_scipy(z_file)

    if 27 == nGLL_per_element:
        glls_idx = ibool[idx_element][CONSTANT_INDEX_27_GLL]
    else:
        glls_idx = ibool[idx_element]

    return x[glls_idx], y[glls_idx], z[glls_idx]



