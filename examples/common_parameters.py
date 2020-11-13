# -------------------------------------------------------------------
# The adjustable parameters used in the inversion.
#
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
n_event_offset        = 60 * sampling_rate      # the offset between the beginning of the downloaded waveform and first arrival in sample.
n_pnl_length          = 30 * sampling_rate   # the length of the cut P segment in sample.
n_pnl_back_offset     = 10 * sampling_rate          # the offset between the beginning of cut P segment and the first P arrival in sample.
n_surface_length      = 120 * sampling_rate  # the length of the cut S segment in sample.
n_surface_back_offset = 20 * sampling_rate          # the offset between the beginning of cut S segment and the first S arrival in sample.
sgt_offset            = 0 # the offset of sgt database expressed in the STF.
n_sgt_offset          = int(sgt_offset * sampling_rate)

cc_threshold               = 0.6
misfit_threshold           = 1.0
timeshift_threshold        = 50
n_valid_waveform_threshold = 1

# filtering parameters
b_pnl_filter = True
p_Tmin       = 5                  # in second.
p_Tmax       = 20                 # in second.
pnl_freq_min = 1.0 / p_Tmax
pnl_freq_max = 1.0 / p_Tmin
b_surface_filter = True
s_Tmin           = 10
s_Tmax           = 20
surface_freq_min = 1.0 / s_Tmax
surface_freq_max = 1.0 / s_Tmin


PROCESSOR   = 1      # The number of the processor used in the inversion.
n_component = 6      # CONSTANT, SHOULD NOT BE CHANGED.

# computing the approximate p and s travel time to cut the SGT (synthetic waveform).
vp = 4.8     # km/s
vs = 3.4     # km/s



# parameters for the second search to find the best focal mechanism.
strike_left  = 5
strike_right = 5
strike_step  = 1
dip_left     = 5
dip_right    = 5
dip_step     = 1

rake_left    = 5
rake_right   = 5
rake_step    = 1

mag_left     = 2.0
mag_right    = 5.0
mag_step     = 0.1


# parameters for the source localization, Searching range.
x_min_range = 0.05
x_max_range = 0.05
y_min_range = 0.05
y_max_range = 0.05
z_min_range = 1000
z_max_range = 500
location_range_model = 'LATLONGZ'
