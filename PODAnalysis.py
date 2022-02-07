import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import xarray as xr
import pickle
from sklearn.decomposition import PCA
# Import library specific modules
from pyspod.spod_low_storage import SPOD_low_storage

snapshots = 100
nx = 256
ny = 128
nz = 160
filename = "/Users/Alex/Desktop/MEng_Resources/2D_3D_DNS_Data/snapshots.pkl"
with open(filename, 'rb') as f:
    obj = pickle.load(f)
    uvw3D_field = obj
data = uvw3D_field[:, :, :, 79, 0]

# u_true = data
# u_true_plot = np.ma.array(u_true)
# u_true_plot[28:49, 45:82] = np.ma.masked
# u_colour = matplotlib.colors.Normalize(-0.50,2.00)
#
# x = np.linspace(0, 16, 256)
# y = np.linspace(-2, 2, 128)
# z = np.linspace(0, 4, 160)
# X, Y = np.meshgrid(x, y, indexing='ij')
#
# fig = plt.figure(figsize=(6*1.8, 1.4*2), dpi=80)
# plt.gca().patch.set_color('.2')
# plt.tick_params(left=False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
# plt.contourf(X,Y,u_true_plot,levels=500, alpha=0.9, cmap='bwr', norm=u_colour)
# cb = plt.colorbar(ticks = [-0.5,0.5,1.5])
# cb.ax.tick_params(labelsize=16)

params = dict()
# -- required parameters
params['dt'] = 1              # data time-sampling
params['nt'] = 100     # number of time snapshots (we consider all data)
params['xdim'] = 2              # number of spatial dimensions
params['nv'] = 1 		# number of variables
params['n_FFT'] = 10         	# length of FFT blocks (100 time-snapshots)

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
spod = SPOD_low_storage(data, params=params, data_handler=False, variables=['u'])

# and run the analysis
spod.fit()

spod.plot_eigs()
