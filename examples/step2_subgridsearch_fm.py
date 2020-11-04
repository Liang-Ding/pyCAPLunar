# -------------------------------------------------------------------
# The script to find the best focal mechanism.
#
# After the fullgridsearch_fm_pnl_s.py, the step will do
# (1) merge the results coming from the grid search.
# (2) search magnitude
# (3) search the precise strike, dip, and rake (subsearch)
# ------- the script stop here --------
#
# The following step is implemented in other code.
# (4) repeat the step (1) to (3) until the magnitude keeps the sample.
# (5) localization.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from examples.common_parameters import proj_result_dir
from examples.tools import find_minimum
from examples.step1_gridsearch_fm import gridsearch_fm

import numpy as np
import sys


def pick_fm(eventid):

    '''
    * Pick the focal mechanism (FM) using the misfit and n_valid_waveform_segment.
    * Select the FM by using the function find_minimum.

    :param eventid: The event index.
    :return:
            strike, dip, rake, mag, misfit, data_segment
    '''

    file_path = str(proj_result_dir) + str('evt_')+str(eventid)+str('/gsfull_fm_pnl_s/') + str('job_gridsearch.txt')
    return find_minimum(file_path)




def get_fm_search_magnitude(eventid, initial_source, strike, dip, rake, mag0, mag1, dmag, job_index='magnitude'):
    '''
    * Get the focal mechanism (FM) by grid search the magnitude after picking the FM from the previous gridsearch.
    * Using grid search but the strike, dip, rake remain unchanged.
    * Only iterate the magnitude.

    :param eventid:    The event index.
    :param strike:      The strike.
    :param dip:         The dip.
    :param rake:        The rake.
    :param mag0:        The minimum magnitude.
    :param mag1:        The maxinum magnitude.
    :param dmag:        The magnitude step.
    :param job_index:   The name of the output file. (job_*.txt)
    :return:
    '''
    strike_0 = strike
    strike_1 = strike + 1
    d_strike = 10
    dip_0 = dip
    dip_1 = dip + 1
    d_dip = 10
    rake_0 = rake
    rake_1 = rake + 1
    d_rake = 10
    mag_0 = mag0
    mag_1 = mag1
    d_mag = dmag
    print("* Search Magnitude ... ")
    gridsearch_fm(eventid=eventid,
                  initial_source=initial_source,
                  strike_0=strike_0,
                  strike_1=strike_1,
                  d_strike=d_strike,
                  dip_0=dip_0,
                  dip_1=dip_1,
                  d_dip=d_dip,
                  rake_0=rake_0,
                  rake_1=rake_1,
                  d_rake=d_rake,
                  mag_0=mag_0,
                  mag_1=mag_1,
                  d_mag=d_mag,
                  job_index=job_index)

    print("* Magnitude-searching has been done!")
    # find the best magnitude
    event_dir = str(proj_result_dir)+ str("evt_") + str(eventid) + str('/')
    mag_search_output_file = str(event_dir) + str('gsfull_fm_pnl_s/job_') +str(job_index)+str('.txt')

    # return the best focal mechanism.
    # strike, dip, rake, mag, misfit, data_segment = find_minimum(mag_search_output_file)
    return find_minimum(mag_search_output_file)



def get_fm_subsearch(eventid, initial_source, strike, round_strike, d_strike,
                      dip, round_dip, d_dip,
                      rake, round_rake, d_rake,
                      mag, round_mag, d_mag, job_index='subsearch'):
    '''
    * Get the best focal mechanism by using second grid search with smaller step and searching range.

    :param eventid:         The event index.
    :param strike:          The roughly determined strike.
    :param round_strike:    The searching range around the strike.
    :param d_strike:        The searching step.
    :param dip:             The roughly determined dip.
    :param round_dip:       The searching range around the dip.
    :param d_dip:           The searching step.
    :param rake:            The roughly determined rake.
    :param round_rake:      The searching range around the dip.
    :param d_rake:          The searching step.
    :param mag:             The roughly determined magnitude.
    :param round_mag:       The searching range around the magnitude.
    :param d_mag:           The searching step.
    :param job_index:       The name of the output file. (job_*.txt)
    :return:
    '''


    # strike
    if strike - round_strike < 180:
        strike_0 = 180
    else:
        strike_0 = strike - round_strike

    if strike + round_strike > 360:
        strike_1 = 360
    else:
        strike_1 = strike + round_strike

    # dip
    if dip - round_dip < 0:
        dip_0 = 0
    else:
        dip_0 = dip - round_dip

    if dip + round_dip > 90:
        dip_1 = 90
    else:
        dip_1 = dip + round_dip

    # rake
    if rake - round_rake < 0:
        rake_0 = 0
    else:
        rake_0 = rake - round_rake

    if rake + round_rake > 360:
        rake_1 = 360
    else:
        rake_1 = rake + round_rake

    # magnitude
    mag_0 = mag - round_mag
    mag_1 = mag + round_mag


    print("Run subsearch!")
    gridsearch_fm(eventid=eventid,
                  initial_source=initial_source,
                  strike_0=strike_0,
                  strike_1=strike_1,
                  d_strike=d_strike,
                  dip_0=dip_0,
                  dip_1=dip_1,
                  d_dip=d_dip,
                  rake_0=rake_0,
                  rake_1=rake_1,
                  d_rake=d_rake,
                  mag_0=mag_0,
                  mag_1=mag_1,
                  d_mag=d_mag,
                  job_index=job_index)

    print("* Sub-searching has been done!")
    # find the best magnitude
    event_dir = str(proj_result_dir) + str("evt_") + str(eventid) + str('/')
    mag_search_output_file = str(event_dir) + str('gsfull_fm_pnl_s/job_') +str(job_index)+str('.txt')

    # return the best focal mechanism.
    # strike, dip, rake, mag, misfit, data_segment = find_minimum(mag_search_output_file)
    return find_minimum(mag_search_output_file)



def determine_fm_for_one(eventid, initial_source):
    '''
    * Search the best focal mechanism of the events.

    :param eventid: The event index.
    :return:
            Write out the best inversion result to the file.
            Format: EVTID, strike, dip, rake, magnitude, misfit, segment.
    '''

    strike0, dip0, rake0, mag0, misfit0, data_segment0 = pick_fm(eventid)

    # parameters for magnitude-gridsearching.
    mag_start = 2.5
    mag_end   = 6.0
    mag_step  = 0.1
    strike1, dip1, rake1, mag1, misfit1, data_segment1 = get_fm_search_magnitude(eventid, initial_source, strike0, dip0, rake0,
                                                                           mag0=mag_start, mag1=mag_end, dmag=mag_step,
                                                                           job_index='magnitude')

    # sub-grid search to find the best (final) focal mechanism.
    round_strike = 2
    d_strike = 1
    round_dip = 2
    d_dip = 1
    round_rake = 2
    d_rake = 1
    round_mag = 0.2
    d_mag = 0.1
    strike_f, dip_f, rake_f, mag_f, misfit_f, data_segment_f = \
        get_fm_subsearch(eventid, initial_source, strike1, round_strike, d_strike,
                         dip1, round_dip, d_dip,
                         rake1, round_rake, d_rake,
                         mag1, round_mag, d_mag, job_index='subsearch')

    # print("results = ", strike_f, dip_f, rake_f, mag_f, misfit_f, data_segment_f)

    # write out the results to the event directory.
    event_dir = str(proj_result_dir) + str("evt_") + str(eventid) + str('/')
    job_index = 'inversion_result'
    result_file = str(event_dir) + str(job_index) + str('.txt')

    with open(result_file, 'w') as f:
        f.write("EVTID, strike, dip, rake, magnitude, misfit, segment\n")
        text_f = str(eventid) + str(', ') \
                 + str(strike_f) + str(', ') \
                 + str(dip_f) + str(',') \
                 + str(rake_f) +str(', ') \
                 + str(mag_f) +str(', ') \
                 + str(misfit_f) + str(', ') \
                 + str(data_segment_f) \
                 + str('\n')
        f.write(text_f)

    return strike_f, dip_f, rake_f, mag_f, misfit_f, data_segment_f

