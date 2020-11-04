# -------------------------------------------------------------------
# An example to calculate the displacement according to the SGT data.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------


from src.DSynthetics.SGT2Disp import SGTs_2_Displacement


def example_sgt_2_displacement()

    sgt_at_any_location = refer_to_example_2_enquire_SGT_data_py__to_get_the_sgt_data()

    return SGTs_2_Displacement(mt=get_moment_tensor(), SGTs=sgt_at_any_location)


if __name__ == '__main__':
    example_sgt_2_displacement()


