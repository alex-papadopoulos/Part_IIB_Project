import pickle
from datetime import datetime
import numpy as np

#Short programme to combine datasets into a single output file

#Store filepaths to be used
file_in = ["/Users/Alex/Desktop/MEng_Resources/2D_3D_DNS_Data/snapshots1.pkl",
           "/Users/Alex/Desktop/MEng_Resources/2D_3D_DNS_Data/snapshots2.pkl"]
file_out = "/Users/Alex/Desktop/MEng_Resources/2D_3D_DNS_Data/snapshots.pkl"

#Load and combine pickle files
with open(file_in[0], 'rb') as f:
    obj = pickle.load(f)
    uvw3D_field = obj
# Flag for first file
print(datetime.now(), "Loaded data from first pickle file")

with open(file_in[1], 'rb') as f:
    obj = pickle.load(f)
    uvw3D_field = np.concatenate((uvw3D_field, obj), axis=0)
# Flag for second file
print(datetime.now(), "Loaded data from second pickle file")

#Output single pickle file
file = open(file_out, 'wb')
pickle.dump(uvw3D_field, file)
file.close()
print(datetime.now(), "Successfully saved combined data file")
