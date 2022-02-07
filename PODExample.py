import os
import xarray as xr
import numpy  as np

# Import library specific modules
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming


# Let's create some 2D syntetic data

# -- define spatial and time coordinates
x1 = np.linspace(0,10,100)
x2 = np.linspace(0, 5, 50)
xx1, xx2 = np.meshgrid(x1, x2)
t = np.linspace(0, 200, 1000)

# -- define 2D syntetic data
s_component = np.sin(xx1 * xx2) + np.cos(xx1)**2 + np.sin(0.1*xx2)
t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
p = np.empty((t_component.shape[0],)+s_component.shape)
for i, t_c in enumerate(t_component):
    p[i] = s_component * t_c


# Let's define the required parameters into a dictionary
params = dict()

# -- required parameters
params['dt'] = 1              # data time-sampling
params['nt'] = t.shape[0]     # number of time snapshots (we consider all data)
params['xdim'] = 2              # number of spatial dimensions
params['nv'] = 1 		# number of variables
params['n_FFT'] = 100          	# length of FFT blocks (100 time-snapshots)

# -- optional parameters
params['n_overlap'] = 0           # dimension block overlap region
params['mean'] = 'blockwise' # type of mean to subtract to the data
params['normalize_weights'] = False       # normalization of weights by data variance
params['normalize_data'] = False       # normalize data by data variance
params['n_modes_save'] = 3           # modes to be saved
params['conf_level'] = 0.95        # calculate confidence level
params['reuse_blocks'] = True        # whether to reuse blocks if present
params['savefft'] = False       # save FFT blocks to reuse them in the future (saves time)
params['savedir'] = os.path.join('results', 'simple_test') # folder where to save results


# Initialize libraries for the low_storage algorithm
spod = SPOD_low_storage(p, params=params, data_handler=False, variables=['p'])

# and run the analysis
spod.fit()


# Let's plot the data
spod.plot_2D_data(time_idx=[1,2])
spod.plot_data_tracers(coords_list=[(5,2.5)], time_limits=[0,t.shape[0]])


# Show results
T_approx = 10 # approximate period = 10 time units
freq = spod.freq
freq_found, freq_idx = spod.find_nearest_freq(freq_required=1/T_approx, freq=freq)
modes_at_freq = spod.get_modes_at_freq(freq_idx=freq_idx)
spod.plot_eigs()
spod.plot_eigs_vs_period(freq=freq, xticks=[1, 7, 30, 365, 1825])
spod.plot_2D_modes_at_frequency(
    freq_required=freq_found, freq=freq, x1=x2, x2=x1, modes_idx=[0,1], vars_idx=[0])