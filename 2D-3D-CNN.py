'''
2D-3D Convolutional Neural Networks
'''

# To run this code, a listed modules are required
from datetime import datetime

import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

# Flag for module import
print(datetime.now(), "Successfully imported modules")

# Parameters
Laptop = False
train_test_split = True
user = "ap2021"
# user = "rpe26"
#filename = "/home/" + user + "/rds/hpc-work/snapshots.pkl"
# filename = ["app/snapshots1.pkl","app/snapshots2.pkl"]
# filename = 'downloads/channel (1).h5'
filename = "/Users/Alex/Desktop/MEng_Resources/2D_3D_DNS_Data/snapshots.pkl"
act = 'relu'
snapshots = 100
nx = 256
ny = 128
nz = 160
EPOCHS = 2000
BATCH_SIZE = 12
VAL_SPLIT = 0.3
time_handling = True
dt = 2  # time delay between slices expressed as number of snapshots
n_slices = 5  # Only supports 3, 5 and 7

if Laptop:
    snapshots = 5
    nx = 64
    ny = 64
    nz = 40
    EPOCHS = 1
    BATCH_SIZE = snapshots
    VAL_SPLIT = 0.2
    
# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))  
    
# Configurations of input/output variables are as follows:
'''
3D_field: time, nx =256, ny=128, nz=160, components=3
'''
if Laptop:

    with open(filename, 'rb') as f:
        obj = pickle.load(f)  # laptop currently only for pickle file
        uvw3D_field = obj[:snapshots, :nx, :ny, :nz, :]  # reduce size
        uvw3D_field = uvw3D_field.astype(np.float32)  # reduce accuracy

    # Flag for first file
    print(datetime.now(), "Loaded data from pickle file for laptop")

else:

    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        uvw3D_field = obj
    # Flag for first file
    print(datetime.now(), "Loaded data from pickle file")

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
else:

    if n_slices == 3:
        # Define empty 2D dataset
        uvw2D_sec = np.empty([snapshots-2*dt, nx, ny, n_slices * 3])
        for i in range(dt, snapshots-dt):
            uvw2D_sec[i - dt, :, :, 0:3] = uvw3D_field[i-dt, :, :, slice_pos[0], :]
            uvw2D_sec[i - dt, :, :, 3:6] = uvw3D_field[i, :, :, slice_pos[1], :]
            uvw2D_sec[i - dt, :, :, 6:9] = uvw3D_field[i + dt, :, :, slice_pos[2], :]

        uvw3D_field = uvw3D_field[dt:snapshots-dt,:,:,:,:]
    elif n_slices == 5:
        # Define empty 2D dataset
        uvw2D_sec = np.empty([snapshots-4*dt, nx, ny, n_slices * 3])
        for i in range(2*dt, snapshots-2*dt):
            uvw2D_sec[i - 2*dt, :, :, 0:3] = uvw3D_field[i-2*dt, :, :, slice_pos[0], :]
            uvw2D_sec[i - 2*dt, :, :, 3:6] = uvw3D_field[i-dt, :, :, slice_pos[1], :]
            uvw2D_sec[i - 2*dt, :, :, 6:9] = uvw3D_field[i, :, :, slice_pos[2], :]
            uvw2D_sec[i - 2*dt, :, :, 9:12] = uvw3D_field[i+dt, :, :, slice_pos[3], :]
            uvw2D_sec[i - 2*dt, :, :, 12:] = uvw3D_field[i+2*dt, :, :, slice_pos[4], :]

        uvw3D_field = uvw3D_field[2*dt:snapshots-2*dt,:,:,:,:]
    elif n_slices == 7:
        # Define empty 2D dataset
        uvw2D_sec = np.empty([snapshots-6*dt, nx, ny, n_slices * 3])
        for i in range(3*dt, snapshots-3*dt):
            uvw2D_sec[i - 3*dt, :, :, 0:3] = uvw3D_field[i-3*dt, :, :, slice_pos[0], :]
            uvw2D_sec[i - 3*dt, :, :, 3:6] = uvw3D_field[i-2*dt, :, :, slice_pos[1], :]
            uvw2D_sec[i - 3*dt, :, :, 6:9] = uvw3D_field[i-dt, :, :, slice_pos[2], :]
            uvw2D_sec[i - 3*dt, :, :, 9:12] = uvw3D_field[i, :, :, slice_pos[3], :]
            uvw2D_sec[i - 3*dt, :, :, 12:15] = uvw3D_field[i+dt, :, :, slice_pos[4], :]
            uvw2D_sec[i - 3*dt, :, :, 15:18] = uvw3D_field[i+2*dt, :, :, slice_pos[5], :]
            uvw2D_sec[i - 3*dt, :, :, 18:] = uvw3D_field[i+3*dt, :, :, slice_pos[6], :]

        uvw3D_field = uvw3D_field[3*dt:snapshots-3*dt,:,:,:,:]



# Flag for dataset shape
print(datetime.now(), "Shape of 3D Dataset", uvw3D_field.shape)

print(datetime.now(), "Shape of 2D dataset", uvw2D_sec.shape)

### train_test_split
if train_test_split:
    from sklearn.model_selection import train_test_split
    print(datetime.now(), "Imported train_test_split")
    uvw2D_sec, uvw2D_sec_val, uvw3D_field, uvw3D_field_val = train_test_split(uvw2D_sec, uvw3D_field, test_size=VAL_SPLIT, shuffle=True)
    print(datetime.now(), "Data split", VAL_SPLIT)

with strategy.scope():
    # Input variables
    input_field = tf.keras.layers.Input(shape=(nx, ny, 3*n_slices))

    # Network structure
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=act, padding='same')(input_field)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=act, padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation=act, padding='same')(x)
    x = tf.keras.layers.Conv2D(int(nz / 8), (3, 3), activation=act, padding='same')(x)
    x = tf.keras.layers.Reshape([int(nx / 2), int(ny / 2), int(nz / 8), 1])(x)
    x = tf.keras.layers.Conv3D(16, (3, 3, 3), activation=act, padding='same')(x)
    x = tf.keras.layers.Conv3D(16, (3, 3, 3), activation=act, padding='same')(x)
    x = tf.keras.layers.UpSampling3D((2, 2, 8))(x)
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), activation=act, padding='same')(x)
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), activation=act, padding='same')(x)
    x_final = tf.keras.layers.Conv3D(3, (3, 3, 3), activation='linear', padding='same')(x)
    # -------- #
    model = tf.keras.models.Model(input_field, x_final)
    model.compile(optimizer='adam', loss='mse')

# Flag for compiling model
print(datetime.now(), "NN Model compiled")

##################

model_cb = tf.keras.callbacks.ModelCheckpoint('/home/' + user + '/rds/hpc-work/Test_Model_Checkpoint.hdf5',
                                              monitor='val_loss', save_best_only=True, verbose=1)
early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)
cb = [model_cb, early_cb]
# Use reduced epochs for first few runs - original is 5000
if train_test_split:
    history = model.fit(uvw2D_sec, uvw3D_field, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=cb, shuffle=True, validation_data=(uvw2D_sec_val,uvw3D_field_val))
else:
    history = model.fit(uvw2D_sec, uvw3D_field, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=cb, shuffle=True, validation_split=VAL_SPLIT)
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='/home/' + user + '/rds/hpc-work/Test_Model_Results.csv', index=False)

model.save("/home/" + user + "/rds/hpc-work/Test_Model")

# Flag for model.save
print(datetime.now(), "Successfully saved trained model")
