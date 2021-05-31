# -------------------------------------------------------------------
# Prepare the SGTs.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import numpy as np
import pickle
import zlib


def DEnquire_SGT(data_path, info_path, GLL_points, encoding_level):
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
            idx_gll = np.where(names_GLL_arr == gll)[0][0]
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
            if 8 == encoding_level:
                data = np.frombuffer(data, dtype=np.uint8)
            else:
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

