# -------------------------------------------------------------------
# Functions to enquire data from the SGT database.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------



from DSEM_Utils import NGLLX, NGLLY, NGLLZ
from DSEM_Utils.ibool_reader import read_ibool_by_scipy
import numpy as np
import h5py
import pickle
import zlib



def DEnquire_SGT_GLL_hdf5(file_path, idx_gll=None):
    '''
    * Enquire the SGT database by the gll point.
    * The SGT is saved at the HDF5 files without compression.

    :param file_path:   The path to the HDF5 file.
                        The SGT size: [n_gll, n_step, n_dim, n_paras]
                        * n_gll: the number of GLL point in one processor
                        * n_step: the number of samples in one trace.
                        * n_dim: constant, the number of unit force.
                        * n_para: constant, the number of the component in SGT.

    :param idx_gll:     The global index of the gll points where the SGT is extracted.
    :return:
            numpy.ndarray, The SGT at specific GLL point.

    '''

    f = h5py.File(file_path, 'r')
    dst = f[list(f.keys())[0]]

    if idx_gll is None:
        return dst
    else:
        try:
            n_gll = len(idx_gll)
            sgt_arr_list = []
            for i in idx_gll:
                sgt_arr_list.append(dst[i, :, :, :])
            return sgt_arr_list

        except:
            return dst[idx_gll, :, :, :]


def DEnquire_SGT_Element_hdf5(sgt_file, ibool_file, idx_element, NSPEC_global):
    '''
    * Enquire the SGT by element.
    * The SGT is saved at the HDF5 files without compression.

    :param sgt_file:        The file path to the HDF5 file.
    :param ibool_file:      The file path of the ibool file. the *_ibool.bin.
    :param idx_element:     The index of the selected element. INT.
    :param NSPEC_global:    The number of element in one processor.
    :return:                The SGT of the sorted GLL points in one element.
    '''

    # read the whole ibool file.
    ibool = read_ibool_by_scipy(ibool_file, NSPEC_global)

    # get the global index array of the GLL points for the selected element.
    idx_arr_gll = ibool[idx_element]
    idx_arr_gll = np.reshape(idx_arr_gll, [NGLLZ, NGLLY, NGLLX])

    # sort the SGT according to the ibool.
    sort_gll = []
    for i in range(NGLLX):
        for j in range(NGLLY):
            for k in range(NGLLZ):
                sort_gll.append(idx_arr_gll[k, j, i])
    sort_gll = np.asarray(sort_gll)

    # extract SGT
    sgt_arr_list = DEnquire_SGT_GLL_hdf5(sgt_file, idx_gll=sort_gll)
    return sgt_arr_list


def DEnquire_SGT(data_path, info_path, GLL_points, encoding_level=8):
    '''
    * Enquire the SGT from the database (*.bin files).

    :param data_path:       The path to the data file (.bin).
    :param info_path:       The path to the info file (.pkl).
    :param GLL_points:      The global index of the GLL points need to be enquired.
    :param encoding_level:  The encoding level.
    :return:
            The list of SGT array enquired.
    '''

    # load the information of SGT database.
    with open(info_path, 'rb') as fr_info:
        n_gll = pickle.load(fr_info)
        n_step = pickle.load(fr_info)
        n_dim = pickle.load(fr_info)
        n_paras = pickle.load(fr_info)
        names_GLL_arr = pickle.load(fr_info)
        data_index_array = pickle.load(fr_info)
        data_offset_array = pickle.load(fr_info)
        data_factor_array = pickle.load(fr_info)

    names_GLL_arr = np.asarray(names_GLL_arr)
    data_index_array = np.asarray(data_index_array)
    data_offset_array = np.asarray(data_offset_array)
    data_factor_array = np.asarray(data_factor_array)

    sgt_arr_list = []
    with open(data_path, "rb") as fr:
        for gll in GLL_points:
            # print("gll=", gll)
            idx_gll = np.where(names_GLL_arr == gll)[0][0]
            # print("index=", idx_gll)

            offset_min = data_offset_array[idx_gll]
            normal_factor = data_factor_array[idx_gll]
            sgt_begin_bytes = data_index_array[idx_gll][0]
            sgt_length_bytes = data_index_array[idx_gll][1]

            # extract the compressed data.
            fr.seek(sgt_begin_bytes)
            data = fr.read(sgt_length_bytes)

            # uncompress the data to bytes.
            data = zlib.decompress(data)

            # recover bytes to uint16
            data = np.frombuffer(data, dtype=np.uint16)

            # recover waveform
            data = data / (2 ** encoding_level - 1) * normal_factor + offset_min

            # data container
            unzip_sgt = np.zeros([n_step, n_dim, n_paras]).astype(np.float32)
            count = 0
            for j in range(n_dim):
                for k in range(n_paras):
                    unzip_sgt[:, j, k] = data[count * n_step:(count + 1) * n_step]
                    count += 1

            sgt_arr_list.append(unzip_sgt)

    return sgt_arr_list


