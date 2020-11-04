# -------------------------------------------------------------------
# The Widgets that make life easier.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import numpy as np


def get_proc_name(idx_processor):
    '''Return the processor name.'''
    if idx_processor < 10:
        str_processor = str('proc00000') + str(idx_processor)
    else:
        if idx_processor < 100:
            str_processor = str('proc0000') + str(idx_processor)
        else:
            if idx_processor < 1000:
                str_processor = str('proc000') + str(idx_processor)
            else:
                if idx_processor < 10000:
                    str_processor = str('proc00') + str(idx_processor)
                else:
                    str_processor = str('proc0') + str(idx_processor)
    return str_processor


