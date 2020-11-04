# -------------------------------------------------------------------
# Common script for reading the binary files exported by the specfem3D.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from scipy.io import FortranFile
import numpy as np


def read_bin_by_scipy(file_path, file_type):
    '''
    * read bin file by using the SciPy package.

    :param file_path: The path of the bin file.
    :param file_type: The data type of the data storing in the Bin file.

    * x,y,z -- float32
    * bin -- int32

    :return: The whole data.
    '''

    try:
        f = FortranFile(file_path, 'r')
        dat = f.read_reals(dtype=file_type)
        f.close()
    except:
        print("Unable to open file: ", str(file_path))
        return None

    return dat



def open_bin(file_path):
    '''Return the file handel which point to a binary file.'''
    try:
        file_handle = open(file_path, "rb")
    except:
        return None
    return file_handle


def close_bin(file_handle):
    '''Close the opened file.'''
    if file_handle is None:
        return
    else:
        file_handle.close()


def read_bin_files(file_handel, L=-1, offset=0, bin_type=np.float32):
    '''
     * Return the bin file.

    :param file_path:   The path of a *.bin file.
    :param L:           The length of data.
    :param offset:      The offset in sample.
    :param bin_type:    The data type, for specfem3D is the float32.
    :return:            The data array read from the bin file.
    '''
    if file_handel is None:
        return None

    bytes_per_sample = 4
    try:
        offset_byte = int(offset * bytes_per_sample)
        file_handel.seek(offset_byte)
        dat = np.fromfile(file_handel, bin_type, int(L))
    except:
        return None

    return dat

