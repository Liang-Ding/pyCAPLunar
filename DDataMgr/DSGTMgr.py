# -------------------------------------------------------------------
# SGT database manager.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from DSEM_Utils.DWidgets import get_proc_name
from DSEM_Utils.ibool_reader import DEnquire_Element
from DSEM_Utils.sgt_reader import DEnquire_SGT
from DSEM_Utils.xyz_reader import DEnquire_XYZ_GLLs_Element
from DDataMgr.DGridMgr import DPolyMesh
from DMath.interp_tools import interp_sgts, DCreate_anchors_xi_eta_gamma, DLagrange_interp_sgt, DLagrange_any3D

import numpy as np

class DSGTMgr(DPolyMesh):

    def __init__(self, sgt_database_dir, model_database_dir,
                 NSPEC_global, encoding_level=8, nGLL_per_element=27,
                 polyMesh_file_path=None):
        '''
        :param sgt_database_dir:    The directory of the sgt database.
        :param model_database_dir:  The directory of the model.
        :param NSPEC_global:        The number of element in one slice (processor).
        :param nGLL_per_element:    The number of GLL in each element saved in the SGT database.
        '''
        self.NSPEC_global = NSPEC_global
        self.nGLL_per_element = int(nGLL_per_element)  # either 27 or 125.
        self.encoding_level = encoding_level

        self.sgt_database_dir = sgt_database_dir
        self.model_database_dir = model_database_dir
        # todo: database checking.

        self.idx_element = -1

        self.b_sgtMgr_initial = False
        super().__init__(polyMesh_file_path)


    def _initial_element_frame(self):
        ''' return the gll information (index, location) at one selected element. '''

        # todo: checking files availability .
        # ibool files.
        ibool_file = str(self.model_database_dir) + self.proc_name + "_ibool.bin"

        # The index of the GLL points
        self.idx_glls = DEnquire_Element(ibool_file, self.NSPEC_global, self.idx_element,
                                         nGLL_per_element=self.nGLL_per_element)
        # The (x, y, z) of the GLL points.
        x_glls, y_glls, z_glls = DEnquire_XYZ_GLLs_Element(self.model_database_dir,
                                                           self.idx_processor, self.idx_element,
                                                           self.NSPEC_global, self.nGLL_per_element)
        self.xyz_glls = np.transpose(np.vstack([x_glls, y_glls, z_glls]))


    def _initial_stations(self, station_names):
        self.station_names = station_names


    def _initial_SGTs_N_station(self):
        '''Return the SGT at selected element for N stations. '''

        self.sgts_element_n_station = []
        for station in self.station_names:
            sgt_data_path = str(self.sgt_database_dir) + str(station) + str('/') + str(self.proc_name) + str(
                "_sgt_data.bin")
            sgt_info_path = str(self.sgt_database_dir) + str(station) + str('/') + str(self.proc_name) + str(
                "_sgt_info.pkl")
            sgts = DEnquire_SGT(data_path=sgt_data_path, info_path=sgt_info_path,
                               GLL_points=self.idx_glls, encoding_level=self.encoding_level)
            self.sgts_element_n_station.append(sgts)


    def Initialize(self, idx_processor, idx_element, station_names):
        '''Initialize'''

        self.idx_processor = idx_processor
        self.proc_name = get_proc_name(self.idx_processor)
        self.idx_element = idx_element

        self._initial_stations(station_names)
        self._initial_element_frame()
        self._initial_SGTs_N_station()
        self.b_sgtMgr_initial = True


    def interp_sgt_linear(self, source):
        ''' Using linear method to interpolate the SGT. '''

        self.sgts_interp_n_station = []
        for sgts in self.sgts_element_n_station:
            sgt_interp = interp_sgts(self.xyz_glls, source, sgts)
            self.sgts_interp_n_station.append(sgt_interp)
        # todo: checking the index processor and element according to the source.


    def interp_sgt_Lagrange(self, xi, eta, gamma):
        '''Using lagrange method to interpolate the SGT at source (in the mesher)'''

        self.sgts_interp_n_station = []
        for i in range(len(self.station_names)):
            xi_gll, eta_gll, gamma_gll = DCreate_anchors_xi_eta_gamma(ngll_xyz=3)
            h_xi_arr, h_eta_arr, h_gamma_arr = DLagrange_any3D(xi, eta, gamma, xi_gll, eta_gll, gamma_gll)
            if 27 == self.nGLL_per_element:
                ngll_x = 3
                ngll_y = 3
                ngll_z = 3
            else:
                ngll_x = 5
                ngll_y = 5
                ngll_z = 5

            sgt_interp = DLagrange_interp_sgt(h_xi_arr, h_eta_arr, h_gamma_arr, self.sgts_element_n_station[i],
                                              ngll_x=ngll_x, ngll_y=ngll_y, ngll_z=ngll_z)
            sgt_interp = sgt_interp[:-1, :, :]
            self.sgts_interp_n_station.append(sgt_interp)


    def get_sgt(self, source, mode='LAGRANGE'):
        '''
        Get the interpolated SGT at the source (lat, long, depth in meter.)
        Eg: source = (36.7, 118.5, -2000)
        '''

        if not self.b_sgtMgr_initial:
            print("Uninitial!")
            return

        if not self.b_polyMesh_initial:
            print("Linear interpolation! As the PolyMesh not initialized!")
            return self.interp_sgt_linear(source)

        _, _, _, \
        _, _, _, \
        idx_processor, element_index, \
        xi, eta, gamma = self.find(x=source[0], y=source[1], z=source[2], n=1)

        # the element index in the denser mesh (coming from the output_solver.txt) starts from 1.
        # MUST subtract 1.
        idx_element = element_index - 1

        # if the source is at other element, re-initial the element and SGTs.
        if idx_processor != self.idx_processor or idx_element != self.idx_element:
            self.idx_processor = idx_processor
            self.idx_element = idx_element
            self.proc_name = get_proc_name(self.idx_processor)
            self._initial_element_frame()
            self._initial_SGTs_N_station()

        # The interpolated SGT
        if 'LAGRANGE' == mode:
            self.interp_sgt_Lagrange(xi, eta, gamma)
        else:
            self.interp_sgt_linear(source)
        return self.sgts_interp_n_station
