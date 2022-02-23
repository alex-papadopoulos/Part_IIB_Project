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
user = "rpe26"
filename = "/home/" + user + "/rds/hpc-work/channel_1_1-100.pkl"
file_out = "/home/" + user + "/rds/hpc-work/channel_linear.pkl"
snapshots = 100
nx = 256
ny = 128
nz = 160
n_slices = 5  # Only supports 3, 5 and 7
time_handling = False

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
    slice_pos[i] = (nz - 1) * (0.5 + slice_distance * (i - (n_slices - 1) / 2))
slice_pos = slice_pos.astype(int)

# Ex. 5 cross-sections are used to input of the model
if not time_handling:
    # Define empty 2D dataset
    uvw2D_sec = np.empty([snapshots, nx, ny, n_slices * 3])

    for i in range(n_slices):
        uvw2D_sec[:, :, :, 3*i: 3*(i+1)] = uvw3D_field[:, :, :, slice_pos[i], :]


###linear interpolate before input to NN
training_field_linear = uvw3D_field.copy()

x = slice_pos
x_new = np.arange(0, nz)
y = predicted_field_linear[:,:,x,:]
f = interpolate.interp1d(x, y, kind='linear', axis= 2, bounds_error=False, fill_value=(y[:,:,0,:],y[:,:,-1,:]))   
predicted_field_linear[:,:,x_new,:] = f(x_new)



# Flag for dataset shape
print(datetime.now(), "Shape of 3D Dataset", uvw3D_field.shape)

print(datetime.now(), "Shape of 3D input dataset", training_field_linear.shape)

#Output single pickle file
file = open(file_out, 'wb')
pickle.dump(training_field_linear, file, protocol=4)
file.close()
print(datetime.now(), "Successfully saved data file")
