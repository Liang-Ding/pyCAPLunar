# -------------------------------------------------------------------
# The working pathes.
#
# exp12
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

# The folder where the observation waveform is placed.
waveform_database_dir =  str('/home/user1/waveform_base') + str('/')

# The folder where the sgt database is stored.
sgt_database_dir = str('/home/user1/sgt_database') + str('/')

# the folder where the *ibool files are stored.
model_database_dir = str('/home/user1/MODEL_DATABASES') + str('/')

# output directory
proj_result_dir = str('/home/user1/example_project') + str('/')

# the follwing parameters
NSPEC_global     = 2232     # The number of element in each slice.
nGLL_per_element = 27       # The number of GLL point in each element stored in SGT database.
n_dim            = 3        # The number of forces.
n_paras          = 6        # The number of element in SGT matrix stored. [xx, yy, zz, xy, xz, yz]
encoding_level   = 8        # The encoding level to compressing the SGT database.

sampling_rate    = 2  # data sampling interval. in Hz
df               = sampling_rate

# parameters for slicing the SGT and waveform.
n_event_offset    = 60 * sampling_rate   # the offset between the begin of the downloaded waveform and the first arrival in sample.
p_waveform_length = 30 * sampling_rate   # the length of the Pnl segment in sample.
p_n_offset        = 10 * sampling_rate   # the offset between the begin of the Pnl segment and the first P arrival in sample.
s_waveform_length = 120 * sampling_rate  # the length of the cut S segment in sample.
s_n_offset        = 20 * sampling_rate   # the offset between the begin of the S/surface segment and the first S arrival in sample.
sgt_offset        = 5.0                  # the offset of sgt database expressed in the STF.
n_sgt_offset      = int(sgt_offset * sampling_rate)

# threshold
cc_threshold             = 0.6
lag_threshold            = 50
misfit_threshold         = 1.0  # relative misfit, (syn - data)/syn * 100
minimum_waveform_segment = 1

# filtering parameters
b_p_filter = True
p_Tmin     = 5                  # in second.
p_Tmax     = 20                 # in second.
p_freqmin  = 1.0 / p_Tmax
p_freqmax  = 1.0 / p_Tmin
b_s_filter = True
s_Tmin     = 10             # in second
s_Tmax     = 20             # in second
s_freqmin  = 1.0 / s_Tmax
s_freqmax  = 1.0 / s_Tmin

#
PROCESSOR   = 1      # The number of the processor used in the inversion.
n_component = 6      # CONSTANT, SHOULD NOT BE CHANGED.

# computing the approximate p and s travel time to cut the SGT (synthetic waveform).
vp = 4.8     # km/s
vs = 3.4     # km/s
