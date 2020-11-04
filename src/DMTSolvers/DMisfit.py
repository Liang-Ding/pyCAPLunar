# -------------------------------------------------------------------
# Moment tensor inversion based on the waveform fitting.
# Compute the waveform misfit between the data and the forward.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------



from DCAPs.DCAPShift import DShift_1d
from DCAPs.DCAPExtend import DExtend_sub
from DCAPs.DCAPUtils import momentMagnitude_2_moment, taper
from Seismology.DFocalTools import DFocalPlane_2_mt
from Seismology.Rotating_tools import DRotate_ENZ2RTZ
from Seismology.geo_tools import gps2dist_azimuth
from DSynthetics.SGT2Disp import SGTs_2_Displacement

import numpy as np


# callable function.
def DMisfit_pnl_s(strike, dip, rake, magnitude,
                sgts_list, data_list,
                loc_src, loc_list_sta,
                n_station, n_component,
                lag_threshold, cc_threshold,
                misfit_threshold,
                minimum_waveform_segment):
    '''

    !!! pnl and surface waveform are separated !!!

    * Return the the waveform misfit between the synthetic and data for n_station.
    * subroutine.
    * Used in the serial (DSolveMT_GridSearch) and concurrent (DSolveMT_GridSearch_MP) version.

    :param strike:      The strike.
    :param dip:         The dip.
    :param rake:        The rake.
    :param magnitude:   The magnitude.
    :param sgts_list:   The sgts data array.
                        Data shape: [[pnl_sgt], [surface_sgt],
                                            ...,
                                    [pnl_sgt], [surface_sgt]]

    :param data_list:   The data array.
                        Data shape: [[pnl_data], [surface_data],
                                            ...,
                                    [pnl_data], [surface_data]]

    :param loc_src:     The location of the source. [Latitude, Longitude]
    :param loc_list_sta:   The list of station position. [[Latitude, Longitude], [Latitude, Longitude], ...,  [Latitude, Longitude]]
    :param n_station:     The number of station.
    :param n_component:   The number of component.

    :param minimum_waveform_segment:    The minimum number of waveform segment that used to calculate the misfit.
                                        Data type: INT.

    :param lag_threshold:               The threshold of the shift (in sample), the maximum time lag (in sample)
                                        Data type: INT
    :param cc_threshold:                The minimum cross-correlation coefficient.
                                        Data type: float < 1.0 (eg: 0.8).

    :param misfit_threshold:            The maximum misfit_threshold.
                                        Data type: float
    :param b_GroupShift:                Compute the time lag and shift the waveform in group.
                                        Set to the False by default.
    :return:
    '''

    # generate moment tensor
    tmp_moment = momentMagnitude_2_moment(np.round(magnitude,2))
    mt_i = DFocalPlane_2_mt(np.deg2rad(strike), np.deg2rad(dip), np.deg2rad(rake), m0=tmp_moment)

    tmp_lags = np.zeros([n_station, n_component])
    tmp_ccs = np.zeros([n_station, n_component])
    tmp_wave_misift = np.zeros([n_station, n_component])
    tmp_weights = np.ones([n_station, n_component])

    # waveform container
    tmp_syn_rtz_list = []
    tmp_data_rtz_list = []


    for i_sta in range(n_station):
        # synthetic waveform in E-N-Z (X-Y-Z) components.
        # pnl and s are in enz.
        pnl_syn_enz = SGTs_2_Displacement(mt_i, sgts_list[i_sta*2])
        pnl_data_enz = data_list[i_sta*2]
        s_syn_enz = SGTs_2_Displacement(mt_i, sgts_list[i_sta*2+1])
        s_data_enz = data_list[i_sta*2+1]

        # compute the back-azimuth according to the source and station position.
        loc_sta = loc_list_sta[i_sta]
        dist, alpha12, alpha21 = gps2dist_azimuth(loc_src[0], loc_src[1], loc_sta[0], loc_sta[1])

        # Rotate E-N-Z to R-T-Z system.
        pnl_syn_rtz = DRotate_ENZ2RTZ(pnl_syn_enz, ba=alpha12)
        s_syn_rtz = DRotate_ENZ2RTZ(s_syn_enz, ba=alpha12)
        pnl_data_rtz = DRotate_ENZ2RTZ(pnl_data_enz, ba=alpha12)
        s_data_rtz = DRotate_ENZ2RTZ(s_data_enz, ba=alpha12)

        # store the waveform by stations, each has 6 waveform segments.
        # syn - pnl, s
        tr_syn_station = []
        tr_data_station = []
        for tr in pnl_syn_rtz:
            tr_syn_station.append(tr)
        for tr in s_syn_rtz:
            tr_syn_station.append(tr)

        # data - pnl, s
        for tr in pnl_data_rtz:
            tr_data_station.append(tr)
        for tr in s_data_rtz:
            tr_data_station.append(tr)

        tmp_syn_rtz_list.append(tr_syn_station)
        tmp_data_rtz_list.append(tr_data_station)

        # if pnl and surface wave are separated. the n_component=6, but get ride of 1th component(P in T component).
        for i_comp in range(n_component):
            # get ride of the P in transverse component.
            if i_comp == 1:
                tmp_weights[i_sta][i_comp] = 0
                continue

            # taper
            tr_syn_station[i_comp] = taper(tr_syn_station[i_comp])
            tr_data_station[i_comp] = taper(tr_data_station[i_comp])

            # Compute the lag and CC for each component individually.
            lag, cc = DShift_1d(tr_syn_station[i_comp], tr_data_station[i_comp])
            tmp_lags[i_sta][i_comp] = lag
            tmp_ccs[i_sta][i_comp] = cc

            '''Compute all misfit'''
            ext_syn_tr = DExtend_sub(tr_syn_station[i_comp], lag)
            ext_wave_tr = DExtend_sub(tr_data_station[i_comp], -1 * lag)

            # Misfit function:
            # Zhu, L., & Helmberger, D. V. (1996). Advancement in source estimation. BSSA, 86(5), 1634–1641
            var_misfit = np.sum(np.square(np.subtract(ext_syn_tr, ext_wave_tr))) / np.dot(np.fabs(ext_syn_tr),
                                                                                          np.fabs(ext_wave_tr))
            tmp_wave_misift[i_sta][i_comp] = var_misfit
            # if the waveform misfit is larger than the threshold, the weight is set to 0.
            if var_misfit > misfit_threshold:
                tmp_weights[i_sta][i_comp] = 0

    n_data_segment = np.count_nonzero(tmp_weights)

    if n_data_segment < minimum_waveform_segment or n_data_segment == 0:
        return 999999.0, strike, dip, rake, magnitude, n_data_segment, \
               tmp_lags, tmp_ccs, tmp_weights, tmp_wave_misift, tmp_syn_rtz_list, tmp_data_rtz_list
    else:
        # compute mean waveform misfit
        return np.mean(np.hstack(tmp_wave_misift[np.nonzero(tmp_weights)])), \
               strike, dip, rake, magnitude, n_data_segment, \
               tmp_lags, tmp_ccs, tmp_weights, tmp_wave_misift, tmp_syn_rtz_list, tmp_data_rtz_list


