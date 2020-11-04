# -------------------------------------------------------------------
# XYZ reader.
# *x.bin, *y.bin, *z.bin
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import sys
sys.path.insert(1, '/scratch/l/liuqy/liangd/workspace/pymt_inversion/')

from DSEM_Utils.DWidgets import get_proc_name
from DSEM_Utils import open_bin, read_bin_files, close_bin
from DSEM_Utils.bin_reader import read_bin_by_scipy
from DSEM_Utils.ibool_reader import read_ibool_by_scipy

import numpy as np



def read_xyz_bins(file_path, L=-1, offset=1, bin_type=np.float32):
    '''
    * Return *(x/y/z).bin file.

    :param file_path:   The path of a *.bin file.
    :param L:           The length of data.
    :param offset:      The offset in sample.
    :param bin_type:    The data type, for specfem3D is the float32.
    :return:            The data array read from the bin file.
    '''

    file_handle = open_bin(file_path)
    # get rid of the first and the last invalid number.
    dat_arr = read_bin_files(file_handle, L=L, offset=offset, bin_type=bin_type)[:-1]
    close_bin(file_handle)
    return dat_arr


def read_xyz_bin_by_scipy(file_path):
    '''
    * read XYZ bin file by using the SciPy package.

    :param file_path: The path of the XYZ bin file.
    :param file_type: The data type of the XYZ bin.

    * x,y,z -- float32

    :return: The whole data.
    '''

    file_type = 'float32'
    return read_bin_by_scipy(file_path, file_type)


def get_xyz_glls_element(data_dir, idx_processor, idx_element, NSPEC_global):
    '''
    * get the x, y, z of the gll points in a specific specific element.

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


    '''
    !!! Attention. 
    The index in ibool is start from 1.
    The index of the gll point is from 0.  
    '''
    glls_idx = ibool[idx_element] - 1

    # read x, y, z
    x = read_xyz_bin_by_scipy(x_file)
    y = read_xyz_bin_by_scipy(y_file)
    z = read_xyz_bin_by_scipy(z_file)

    # get x, y, z
    x_glls = x[glls_idx]
    y_glls = y[glls_idx]
    z_glls = z[glls_idx]

    return x_glls, y_glls, z_glls


