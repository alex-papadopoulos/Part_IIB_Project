'''
2D-3D Convolutional Neural Networks
'''

# To run this code, a listed modules are required
from datetime import datetime

import pickle
import numpy as np
import pandas as pd
from scipy import interpolate

# Flag for module import
print(datetime.now(), "Successfully imported modules")

# Parameters
# user = "ap2021"
user = "ap2021"
# filename = "/home/" + user + "/rds/hpc-work/channel_1_1-100.pkl"
# file_out = "/home/" + user + "/rds/hpc-work/channel_linear.pkl"
filename = "/Users/Alex/Desktop/MEng_Resources/2D_3D_DNS_Data/snapshots.pkl"
file_out = "/Users/Alex/Desktop/MEng_Resources/Project_Run_Logbook/linear_int_xscan_dt5"
snapshots = 100
nx = 256
ny = 128
nz = 160
n_slices = 5  # Only supports 3, 5 and 7
dt = 5
time_handling = True

# Configurations of input/output variables are as follows:
#'''
#3D_field: time, nx =256, ny=128, nz=160, components=3
#...

with open(filename, 'rb') as f:
    obj = pickle.load(f)
    uvw3D_field = obj
# Flag for first file
print(datetime.now(), "Loaded data from pickle file")


####
slice_pos = np.zeros(n_slices)
slice_distance = 0.8/ (n_slices - 1)
for i in range(n_slices):
    slice_pos[i] = (nx - 1) * (0.5 + slice_distance * (i - (n_slices - 1) / 2))
slice_pos = slice_pos.astype(int)

# Ex. 5 cross-sections are used to input of the model
if not time_handling:
    # Define empty 2D dataset
    uvw2D_sec = np.empty([snapshots, ny, nz, 3*n_slices])

    uvw2D_sec[:, :, :, 0:3] = uvw3D_field[:, slice_pos[0], :, :, :]
    uvw2D_sec[:, :, :, 3:6] = uvw3D_field[:, slice_pos[1], :, :, :]
    uvw2D_sec[:, :, :, 6:9] = uvw3D_field[:, slice_pos[2], :, :, :]
    uvw2D_sec[:, :, :, 9:12] = uvw3D_field[:, slice_pos[3], :, :, :]
    uvw2D_sec[:, :, :, 12:] = uvw3D_field[:, slice_pos[4], :, :, :]

else:
    # Define empty 2D dataset
    uvw2D_sec = np.empty([snapshots-4*dt, ny, nz, 15])

    for i in range(2*dt, snapshots-2*dt):
        uvw2D_sec[i - 2*dt, :, :, 0:3] = uvw3D_field[i-2*dt, slice_pos[0], :, :, :]
        uvw2D_sec[i - 2*dt, :, :, 3:6] = uvw3D_field[i-dt, slice_pos[1], :, :, :]
        uvw2D_sec[i - 2*dt, :, :, 6:9] = uvw3D_field[i, slice_pos[2], :, :, :]
        uvw2D_sec[i - 2*dt, :, :, 9:12] = uvw3D_field[i+dt, slice_pos[3], :, :, :]
        uvw2D_sec[i - 2*dt, :, :, 12:] = uvw3D_field[i+2*dt, slice_pos[4], :, :, :]


uvw3D_field = uvw3D_field[2*dt:snapshots-2*dt,:,:,:,:]

uvw3D_field = np.transpose(uvw3D_field, (0,2,3,1,4))


###linear interpolate before input to NN
full_snapshot = uvw3D_field[49,:,:,:,:]
y = np.zeros(full_snapshot[:,:,slice_pos,:].shape)

x = slice_pos
x_new = np.arange(0, nx)
for i in range(len(x)):
    y[:, :, i, :] = uvw3D_field[49-(i-2)*dt,:,:,x[i],:]
f = interpolate.interp1d(x, y, kind='linear', axis= 2, bounds_error=False, fill_value=(y[:,:,0,:],y[:,:,-1,:]))
full_snapshot[:,:,x_new,:] = f(x_new)

# Reorder snapshot dimensions
full_snapshot = np.transpose(full_snapshot, (2,0,1,3))

# Flag for dataset shape
print(datetime.now(), "Shape of 3D Dataset", uvw3D_field.shape)

print(datetime.now(), "Shape of 3D input dataset", full_snapshot.shape)

#Output single pickle file
file = open(file_out, 'wb')
pickle.dump(full_snapshot, file, protocol=4)
file.close()
print(datetime.now(), "Successfully saved data file")
