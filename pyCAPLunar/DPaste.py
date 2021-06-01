# -------------------------------------------------------------------
# Paste the synthetic waveform (syn) on the data.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from pyCAPLunar.DMisfit import DMisfit
import numpy as np


def DPaste(data, n_t_data, n_data_lens, syn, n_t_syn, n_syn_lens, n_max_shift):
    '''
    Paste the synthetic waveform on the data,
    The syn is sliced into phases while the data keeps the original traces.

    ****
    * The data are expected to be longer than the syn.
    ****

    :param data:        The data trace. 1C.
    :param n_t_data:    The index indicating the approximate arrival time of a phase. INT
    :param n_data_lens: The length of the data. INT.
    :param syn:         The synthetic waveform of the phase. 1C.
    :param n_t_syn:     The approximate arrival time of the phase on syn. waveform. INT.
    :param n_syn_lens   The length of the synthetic wave. INT.
    :param n_max_shift: The maximum allowing misfit. INT.
    :return:
            n_shift: the time shift.
            misfit: the misfit.
    '''

    n_max_shift_2 = int(2.0 * n_max_shift)+1

    # compute tiem shift
    i_shift = 0
    _var_cc = 0
    max_cc = 0
    for i in range(n_max_shift_2):
        # idx on the data
        idx = n_t_data - n_t_syn - n_max_shift + i
        if idx + n_syn_lens <= n_data_lens:
            if idx < 0:
                _var_cc = np.dot(syn, np.hstack([np.zeros((-1 * idx)), data[:idx + n_syn_lens]]))
            else:
                _var_cc = np.dot(syn, data[idx:idx + n_syn_lens])
        else:
            if idx < 0:
                _var_cc = np.dot(syn, np.hstack([np.zeros(-1 * idx), data, np.zeros(n_syn_lens + idx - n_data_lens)]))
            else:
                _var_cc = np.dot(syn, np.hstack([data[idx:], np.zeros(n_syn_lens - (n_data_lens - idx))]))
        if 0 == i:
            max_cc = _var_cc
        else:
            if max_cc < _var_cc:
                max_cc = _var_cc
                i_shift = i
    n_shift = i_shift - n_max_shift

    # calculate the waveform misfit and cc.
    n0_idx = n_t_data - n_t_syn + n_shift
    if n0_idx >= 0:
        if n0_idx + n_syn_lens > n_data_lens:
            tmp = data[n0_idx:]
            misfit = DMisfit(syn, np.hstack([tmp, np.zeros(n_syn_lens-len(tmp))]))
            cc = np.corrcoef(syn, np.hstack([tmp, np.zeros(n_syn_lens-len(tmp))]))[0][1]
        else:
            misfit = DMisfit(syn, data[n0_idx:n0_idx+n_syn_lens])
            cc = np.corrcoef(syn, data[n0_idx:n0_idx+n_syn_lens])[0][1]
    else:
        if n0_idx + n_syn_lens > n_data_lens:
            tmp = np.hstack([np.zeros(-1 * n0_idx), data])
            misfit = DMisfit(syn, np.hstack([tmp, np.zeros(n_syn_lens - len(tmp))]))
            cc = np.corrcoef(syn, np.hstack([tmp, np.zeros(n_syn_lens - len(tmp))]))[0][1]
        else:
            misfit = DMisfit(syn, np.hstack([np.zeros(-1 * n0_idx), data[:n_syn_lens + n0_idx]]))
            cc = np.corrcoef(syn, np.hstack([np.zeros(-1 * n0_idx), data[:n_syn_lens + n0_idx]]))[0][1]

    return int(n_shift), misfit, cc

