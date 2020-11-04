# -------------------------------------------------------------------
# Example of enquiring data form the SGT database.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from examples.common_parameters import model_database_dir, sgt_database_dir, NSPEC_global, nGLL_per_element

from src.DMath.interp_tools import DCreate_anchors_xi_eta_gamma, DLagrange_interp_sgt, DLagrange_any3D
from src.DSEM_Utils.DWidgets import get_proc_name
from src.DSEM_Utils.ibool_reader import DEnquire_Element
from src.DSEM_Utils.sgt_reader import DEnquire_SGT

import numpy as np


def example_enquire_sgt_data():
    ''' Enquire the SGT at an arbitrarily selected location'''

    idx_processor = 134
    proc_name = get_proc_name(idx_processor)
    idx_element = 632 - 1  # MUST subtract 1. the index of the element starts from 1.

    ibool_file = str(model_database_dir) + proc_name + "_ibool.bin"

    station_name = 'VTV'
    data_path = str(sgt_database_dir) + str(station_name) + str('/') + str(proc_name) + str("_sgt_data.bin")
    info_path = str(sgt_database_dir) + str(station_name) + str('/') + str(proc_name) + str("_sgt_info.pkl")

    # 1. Enquire the GLL index.
    gll_points = DEnquire_Element(ibool_file, NSPEC_global, idx_element, nGLL_per_element=nGLL_per_element)

    # 2. Enquire the SGT.
    sgt_arr_list = DEnquire_SGT(data_path=data_path, info_path=info_path, GLL_points=gll_points)

    # 3. Interpolation to get the SGT at an arbitrarily selected location.
    xi_gll, eta_gll, gamma_gll = DCreate_anchors_xi_eta_gamma(ngll_xyz=3)

    # the xi, eta, and gamma of the arbitrarily selected location.
    xi_xi = -0.69216636470196558
    eta_yi = -0.16980945231847744
    gamma_zi = 0.48271687393010160
    h_xi_arr, h_eta_arr, h_gamma_arr = DLagrange_any3D(xi_xi, eta_yi, gamma_zi, xi_gll, eta_gll, gamma_gll)

    ngll_x = 3
    ngll_y = 3
    ngll_z = 3

    return DLagrange_interp_sgt(h_xi_arr, h_eta_arr, h_gamma_arr, sgt_arr_list, ngll_x=ngll_x, ngll_y=ngll_y,
                                ngll_z=ngll_z)


if __name__ == '__main__':
    example_enquire_sgt_data()



