# -------------------------------------------------------------------
# Prepare the SGTs.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


import numpy as np
import h5py
import zlib


SGT_KEYS = [
    'index',
    'start',
    'length'
    'offset',
    'scale'
]

SGT_ATTRS = [
    'ngll',
    'nstep',
    'nforce',
    'nparas',
]


def DEnquire_SGT(data_path, info_path, GLL_points, encoding_level):
    '''
    * Enquire the SGT from the database (*.bin files).

    :param data_path:       The path to the data file (.bin).
    :param info_path:       The path to the header file (.hdf5).
    :param GLL_points:      The global index of the GLL points need to be enquired.
    :param encoding_level:  The encoding level.
    :return:
            The list of SGT array enquired.
    '''

    # load the information of SGT database.
    with h5py.File(info_path, 'r') as f:
        names_GLL_arr       = f[SGT_KEYS[0]][:]
        data_start_array    = f[SGT_KEYS[1]][:]
        data_length_array   = f[SGT_KEYS[2]][:]
        data_offset_array   = f[SGT_KEYS[3]][:]
        data_scale_array    = f[SGT_KEYS[4]][:]

        n_gll   = f.attrs[SGT_ATTRS[0]]
        n_step  = f.attrs[SGT_ATTRS[1]]
        n_dim   = f.attrs[SGT_ATTRS[2]]
        n_paras = f.attrs[SGT_ATTRS[3]]


    sgt_arr_list = []
    with open(data_path, "rb") as fr:
        for gll in GLL_points:
            idx_gll = (np.where(names_GLL_arr == gll))[0][0]

            offset_min = data_offset_array[idx_gll]
            normal_factor = data_scale_array[idx_gll]
            sgt_begin_bytes = data_start_array[idx_gll]
            sgt_length_bytes = data_length_array[idx_gll]

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
