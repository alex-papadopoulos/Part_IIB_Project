import pickle
from datetime import datetime
import numpy as np
import h5py

#Short programme to combine datasets into a single output file

#Store filepaths to be used
file_in_1 = "1-10/channel.h5"
file_in = ["11-20/channel.h5","21-30/channel.h5","31-40/channel.h5","41-70/channel.h5","71-100/channel.h5"]
file_out = "snapshots_channel_240122.pkl"

#Load and combine h5py files

f = h5py.File(file_in_1, 'r')
keys = list(f.keys())
array = np.ndarray(shape=(len(keys) - 3, 160, 128, 256, 3))
i = 0
for x in keys[:-3:]:
    dset_temp = f[x]
    array[i, :, :, :, :] = dset_temp
    i += 1
uvw3D_field = np.swapaxes(array, 1, 3)
print(datetime.now(), "Loaded data from h5py file")



for filename in file_in:

  f = h5py.File(filename, 'r')
  keys = list(f.keys())
  array = np.ndarray(shape=(len(keys) - 3, 160, 128, 256, 3))
  i = 0
  for x in keys[:-3:]:
      dset_temp = f[x]
      array[i, :, :, :, :] = dset_temp
      i += 1
  uvw3D_field = np.concatenate((uvw3D_field, np.swapaxes(array, 1, 3)), axis=0)
  print(datetime.now(), "Loaded data from h5py file")



#Output single pickle file
file = open(file_out, 'wb')
pickle.dump(uvw3D_field, file)
file.close()
print(datetime.now(), "Successfully saved combined data file")
