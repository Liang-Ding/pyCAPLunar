# -------------------------------------------------------------------
# An example to store the SGT to files.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from examples.common_parameters import model_database_dir, sgt_database_dir, NSPEC_global, nGLL_per_element
from src.DSEM_Utils.merge_strainfield import DMerge_and_Compress_SGT
from src.DSEM_Utils.ibool_reader import DEquire_ibool
from src.DSEM_Utils.DWidgets import get_proc_name
import numpy as np
import os


def example_merge_and_compress_SGT_database():
    # the '/' should be added at the end of each directory.
    dir_array = ['the directory where the *strain_field*.bin is saved when using the NORTHEN unit force/',
                 'the directory where the *strain_field*.bin is saved when using the EASTERN unit force/',
                 'the directory where the *strain_field*.bin is saved when using the UPWARDS unit force/']

    # select the subarea (in slice) where the SGT is going to be stored.
    # the index of slice starts from 0.
    procs = np.array([119, 120, 133, 134, 147, 148], dtype=np.int)

    ######################################################
    # The parameter is set manually.
    # Later it will be update.

    # the first index in the file of *strain_field_Step*.bin
    step0 = 1
    # the last index in the file of *strain_field_Step*.bin
    step1 = 18002
    # the step interval.
    dstep = 50
    ######################################################

    # station
    station_name = 'The station name, for example VTV'

    # generate the proc name.
    for pc in procs:
        proc_name = get_proc_name(pc)

        # 1. Read the ibool file and get the enquired global index of GLL points.
        ibool_file = str(model_database_dir) + proc_name + "_ibool.bin"

        # if n_GLL_per_element = 125, no spatial sub-sampling.
        names_GLL_arr, index_GLL_arr = DEquire_ibool(ibool_file, NSPEC_global, nGLL_per_element=nGLL_per_element)

        save_dir = str(sgt_database_dir) + str(station_name) + str('/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 2. Merge, compress, and store the SGT database at enquired GLL points.
        DMerge_and_Compress_SGT(dir_array, proc_name, NSPEC_global, names_GLL_arr, index_GLL_arr,
                                step0, step1, dstep, save_dir, n_dim=3, n_paras=6, encoding_level=8)

if __name__ == '__main__':
    example_merge_and_compress_SGT_database()


