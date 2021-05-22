# -------------------------------------------------------------------
# Data Manager
# PolyMesh that stores grid location with interpolating polynomials.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import numpy as np
import pickle


class DPolyMesh():
    '''Class to operate the mesh grid. '''

    def __init__(self, file_path):
        self.b_polyMesh_initial = False
        if file_path is None:
            return

        try:
            with open(file_path, 'rb') as f:
                _ = pickle.load(f)
                self.mesh_lat = pickle.load(f)
                self.mesh_long = pickle.load(f)
                self.mesh_z = pickle.load(f)        # in meter. the depth is minus.
                self.mesh_utm_x = pickle.load(f)
                self.mesh_utm_y = pickle.load(f)
                self.mesh_utm_z = pickle.load(f)
                self.mesh_slice_index = pickle.load(f)
                self.mesh_element_index = pickle.load(f)
                self.mesh_xi = pickle.load(f)
                self.mesh_eta = pickle.load(f)
                self.mesh_gamma = pickle.load(f)

            self.n_grid = len(self.mesh_lat)
            self.b_polyMesh_initial = True
        except:
            print("Unable to initial PolyMesh!")


    def _check(self):
        if self.b_polyMesh_initial is False:
            return [], [], [], [], [], [], [], [], [], [], []


    def find(self, x, y, z, n, mode='LATLONGZ'):
        '''
        * Find n point that near the (x, y, z) from the PolyMesh.

         :param x:       Either the latitude or UTMX.
         :param y:       Either the longitude or UTMY.
         :param z:       Eithe the depth or elevation in meter. (downward is negative).
         :param n:       The number of point enquired.
         :return:        The information of the inquired grid points.
        '''

        self._check()

        n = int(n)
        if str(mode).upper() == str('LATLONGZ'):
            dist = 111.0 * 1000.0 * np.sqrt(np.square(self.mesh_lat - x) + np.square(self.mesh_long - y))
            distance_arr = np.sqrt(np.power(dist, 2) + np.power(self.mesh_z - z, 2))
        elif str(mode).upper() == str('UTM'):
            distance_arr = np.sqrt(
                np.power(self.mesh_utm_x - x, 2) + np.power(self.mesh_utm_y - y, 2) + np.power(self.mesh_utm_z - z, 2))
        else:
            print("Undefined mode!!")
            exit(-1)

        if 1 == n:
            idx = np.argmin(distance_arr)
        else:
            idx = np.argpartition(distance_arr, n)[:n]

        return self.mesh_lat[idx], self.mesh_long[idx], self.mesh_z[idx], \
               self.mesh_utm_x[idx], self.mesh_utm_y[idx], self.mesh_utm_z[idx], \
               self.mesh_slice_index[idx], self.mesh_element_index[idx], \
               self.mesh_xi[idx], self.mesh_eta[idx], self.mesh_gamma[idx]



    def find_cubic(self, x_min, x_max, y_min, y_max, z_min, z_max, mode='LATLONGZ'):
        '''
        Enquire the information of the grid point that in a specific range.

        :param x_min, x_max:    The range [x_min, x_max] in X.  Either the latitude or UTMX.
        :param y_min, y_max:    The range [y_min, y_max] in Y.  Either the longitude or UTMY.
        :param z_min, z_max:    The range [z_min, z_max] in Z.  Eithe the depth or elevation in meter. (downward is negative).
        :param mode:            The searching mode. Either the 'LATLONGZ' or 'UTM'
        :return:                The information of the grid in the inquired cubic.
        '''

        self._check()

        idx = []
        if str(mode).upper() == str('LATLONGZ'):
            for i in range(self.n_grid):
                if (self.mesh_lat[i] >= x_min) & (self.mesh_lat[i] <= x_max) & \
                        (self.mesh_long[i] >= y_min) & (self.mesh_long[i] <= y_max) & \
                        (self.mesh_z[i] >= z_min) & (self.mesh_z[i] <= z_max):
                    idx.append(i)

        elif str(mode).upper() == str('UTM'):
            for i in range(self.n_grid):
                if (self.mesh_utm_x[i] >= x_min) & (self.mesh_utm_x[i] <= x_max) & \
                        (self.mesh_utm_y[i] >= y_min) & (self.mesh_utm_y[i] <= y_max) & \
                        (self.mesh_utm_z[i] >= z_min) & (self.mesh_utm_z[i] <= z_max):
                    idx.append(i)
        else:
            print("Undefined mode!!")
            exit(-1)

        return self.mesh_lat[idx], self.mesh_long[idx], self.mesh_z[idx], \
               self.mesh_utm_x[idx], self.mesh_utm_y[idx], self.mesh_utm_z[idx], \
               self.mesh_slice_index[idx], self.mesh_element_index[idx], \
               self.mesh_xi[idx], self.mesh_eta[idx], self.mesh_gamma[idx]

