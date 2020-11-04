# -------------------------------------------------------------------
# An example to do the inversion by the grid search.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from examples.step1_gridsearch_fm import gridsearch_fm
from examples.step2_subgridsearch_fm import determine_fm_for_one
from examples.step3_localization import error_by_location, find_location

import numpy as np

def event_inversion():
    eventid = 11058623
    initial_source = np.array([36.6758, -1174853, 8530])
    strike_0 = 180.0
    strike_1 = 360.0
    d_strike = 5.0
    dip_0 = 0.0
    dip_1 = 90.0
    d_dip = 3.0
    rake_0 = 0
    rake_1 = 360
    d_rake = 5.0
    mag_0 = 1.0
    mag_1 = 7.0
    d_mag = 0.1

    # first grid search
    print("! Doing grid-search ... ")
    gridsearch_fm(eventid, initial_source,
                  strike_0, strike_1, d_strike,
                  dip_0, dip_1, d_dip,
                  rake_0, rake_1, d_rake,
                  mag_0, mag_1, d_mag)

    print("! Doing second grid-search ...")
    # second grid search to find the best focal mechanism using the smaller searching step.
    strike_f, dip_f, rake_f, mag_f, misfit_f, data_segment_f = determine_fm_for_one(eventid, initial_source)

    print("! Doing inversion ... ")
    # source localization.
    n_point = 100
    error_by_location(eventid, initial_source, strike_f, dip_f, rake_f, mag_f, n_point)

    # find the results
    lat_res, long_res, z_res, strike_res, dip_res, rake_res, mag_res, misfit_res, n_segment_res = find_location(eventid)
    print("! Inversion results:")
    print("Location: [{}, {}, {}].".format(lat_res, long_res, z_res))
    print("Focal mechanism: strike={}, dip={}, rake={}, magnitude={}.".format(strike_res, dip_res, rake_res, mag_res))
    print("{} waveform segments are selected and the inversion misfit is {}. ".format(n_segment_res, misfit_res))
    print("! DONE.")


if __name__ == '__main__':
    event_inversion()
