# -------------------------------------------------------------------
# Functions to read the *strain_field*.bin files.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from DSEM_Utils import NGLLX, NGLLY, NGLLZ
from DSEM_Utils.bin_reader import open_bin, read_bin_files, close_bin
import numpy as np


def read_strain_bins(file_path, NSPEC_global, bin_type=np.float32):
    '''
    * Return the strain field export by the Specfem3D dl_runtime_saver.f90
    *
    * - ATTENTION:
    * The code for computing the deviatory strain (the epsilondev)
      in src/specfem3D/compute_force_viscoelastic.f90 is:
        ...
        epsilondev_trace_loc = duxdxl + duydyl + duzdzl
        epsilondev_xx_loc(i,j,k) = duxdxl - epsilondev_trace_loc
        epsilondev_yy_loc(i,j,k) = duydyl - epsilondev_trace_loc
       ...
    * - To get accurate epsilon.
        epsilondev_xx = epsilondev_xx + epsilondev_trace
        epsilondev_yy = epsilondev_yy + epsilondev_trace
        epsilondev_zz = epsilondev_trace - epsilondev_xx - epsilondev_yy


    :param file_path:       The file path of the bin file.
    :param NSPEC_global:    The total number of elements
    :param bin_type:        The data type, usually is single-precision float(32 bit).
    :return:        The epsilon field (Exx, Eyy, Ezz, Exy, Exz, Eyz)
    '''

    # The data is placed as the order:
    # [N/A][D-A-T-A][N/A] [N/A][D-A-T-A][N/A] [N/A][D-A-T-A][N/A] ...
    # To get rid of the first N/A
    offset = 1
    # To get ride of the N/A between two data segments.
    inter_offset = 2

    file_handle = open_bin(file_path)
    data_length = NGLLX * NGLLY * NGLLZ * NSPEC_global

    epsilondev_trace = read_bin_files(file_handle, L=data_length, offset=offset, bin_type=bin_type)
    epsilondev_xx = read_bin_files(file_handle, L=data_length, offset=offset + data_length + inter_offset, bin_type=bin_type)
    epsilondev_yy = read_bin_files(file_handle, L=data_length, offset=offset + 2 * (data_length + inter_offset), bin_type=bin_type)
    epsilondev_xy = read_bin_files(file_handle, L=data_length, offset=offset + 3 * (data_length + inter_offset), bin_type=bin_type)
    epsilondev_xz = read_bin_files(file_handle, L=data_length, offset=offset + 4 * (data_length + inter_offset), bin_type=bin_type)
    epsilondev_yz = read_bin_files(file_handle, L=data_length, offset=offset + 5 * (data_length + inter_offset), bin_type=bin_type)

    epsilondev_xx = epsilondev_xx + epsilondev_trace / 3.0
    epsilondev_yy = epsilondev_yy + epsilondev_trace / 3.0
    epsilondev_zz = epsilondev_trace - epsilondev_xx - epsilondev_yy

    close_bin(file_handle)

    return np.vstack([epsilondev_xx,
                      epsilondev_yy,
                      epsilondev_zz,
                      epsilondev_xy,
                      epsilondev_xz,
                      epsilondev_yz])


def read_strain_bins_by_elements(file_path, NSPEC_global, bin_type=np.float32):
    '''
    * Return the strain field export by the Specfem3D dl_runtime_saver.f90
    * as elements
    *
    * - ATTENTION:
    * The code for computing the deviatory strain (the epsilondev)
      in src/specfem3D/compute_force_viscoelastic.f90 is :
        ...
        epsilondev_trace_loc = duxdxl + duydyl + duzdzl
        epsilondev_xx_loc(i,j,k) = duxdxl - epsilondev_trace_loc
        epsilondev_yy_loc(i,j,k) = duydyl - epsilondev_trace_loc
       ...
    * - To get accurate epsilon.
        epsilondev_xx = epsilondev_xx + epsilondev_trace
        epsilondev_yy = epsilondev_yy + epsilondev_trace
        epsilondev_zz = epsilondev_trace - epsilondev_xx - epsilondev_yy


    :param file_path:       The file path of the bin file.
    :param NSPEC_global:    The total number of elements
    :param bin_type:        The data type, usually is single-precision float(32 bit).
    :return:        The epsilon field list in element,
                    data size = n_para * NSPEC_global * NGLL
                    n_para= 6, constant, (Exx, Eyy, Ezz, Exy, Exz, Eyz)
    '''

    # The data is placed as the order:
    # [N/A][D-A-T-A][N/A] [N/A][D-A-T-A][N/A] [N/A][D-A-T-A][N/A] ...
    # To get rid of the first N/A
    offset = 1
    # To get ride of the N/A between two data segments.
    inter_offset = 2

    file_handle = open_bin(file_path)
    data_length = NGLLX * NGLLY * NGLLZ * NSPEC_global

    epsilondev_trace = read_bin_files(file_handle, L=data_length, offset=offset, bin_type=bin_type)
    epsilondev_xx = read_bin_files(file_handle, L=data_length, offset=offset + data_length + inter_offset, bin_type=bin_type)
    epsilondev_yy = read_bin_files(file_handle, L=data_length, offset=offset + 2 * (data_length + inter_offset), bin_type=bin_type)
    epsilondev_xy = read_bin_files(file_handle, L=data_length, offset=offset + 3 * (data_length + inter_offset), bin_type=bin_type)
    epsilondev_xz = read_bin_files(file_handle, L=data_length, offset=offset + 4 * (data_length + inter_offset), bin_type=bin_type)
    epsilondev_yz = read_bin_files(file_handle, L=data_length, offset=offset + 5 * (data_length + inter_offset), bin_type=bin_type)

    epsilondev_xx = epsilondev_xx + epsilondev_trace / 3.0
    epsilondev_yy = epsilondev_yy + epsilondev_trace / 3.0
    epsilondev_zz = epsilondev_trace - epsilondev_xx - epsilondev_yy

    close_bin(file_handle)

    strain_list = []
    strain_list.append(np.reshape(epsilondev_xx, (NSPEC_global, NGLLX * NGLLY * NGLLZ)))
    strain_list.append(np.reshape(epsilondev_yy, (NSPEC_global, NGLLX * NGLLY * NGLLZ)))
    strain_list.append(np.reshape(epsilondev_zz, (NSPEC_global, NGLLX * NGLLY * NGLLZ)))
    strain_list.append(np.reshape(epsilondev_xy, (NSPEC_global, NGLLX * NGLLY * NGLLZ)))
    strain_list.append(np.reshape(epsilondev_xz, (NSPEC_global, NGLLX * NGLLY * NGLLZ)))
    strain_list.append(np.reshape(epsilondev_yz, (NSPEC_global, NGLLX * NGLLY * NGLLZ)))

    # the returning data size = 6 * NSPEC_Global * NGLL
    return np.asarray(strain_list)
    
    
