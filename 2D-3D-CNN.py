'''
2D-3D Convolutional Neural Networks
'''

# To run this code, a listed modules are required
from datetime import datetime

import tensorflow as tf
# from keras.layers import Input,Conv2D, Conv3D, MaxPooling2D, UpSampling3D, Reshape
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
# from keras import backend as K
# from keras.callbacks import TensorBoard
# from keras.backend import tf as ktf
import pickle
import numpy as np
import pandas as pd

# Flag for module import
print(datetime.now(), "Successfully imported modules")


# Parameters
Laptop = False
filename = ["/home/ap2021/rds/hpc-work/snapshots1.pkl","/home/ap2021/rds/hpc-work/snapshots2.pkl"]
#filename = ["app/snapshots1.pkl","app/snapshots2.pkl"]
#filename = 'downloads/channel (1).h5'
savelocation = [''] ##not used 
filetype = "pickle"  ###pickle or h5py
act = 'relu'
snapshots = 100
nx =256
ny=128
nz=160
EPOCHS = 100
BATCH_SIZE = 50

###add a parameter to change number of 2D slices? currently 5

if Laptop:
  snapshots = 5
  nx = 64
  ny = 64
  nz = 40
  EPOCHS = 1
  BATCH_SIZE = snapshots

# Configurations of input/output variables are as follows:
'''
3D_field: time, nx =256, ny=128, nz=160, components=3
'''
if Laptop:
  
  with open(filename[0], 'rb') as f:
    obj = pickle.load(f)                         ###laptop currently only for pickle file
    uvw3D_field = obj[:snapshots,:nx,:ny,:nz,:]  ###reduce size
    uvw3D_field = uvw3D_field.astype(np.float32) ###reduce accuracy  
  # Flag for first file
  print(datetime.now(), "Loaded data from pickle file for laptop")
  
elif (filetype == "pickle"):
  
  with open(filename[0], 'rb') as f:
      obj = pickle.load(f)
      uvw3D_field = obj
  # Flag for first file
  print(datetime.now(), "Loaded data from first pickle file")

  with open(filename[1], 'rb') as f:
      obj = pickle.load(f)
      uvw3D_field = np.concatenate((uvw3D_field, obj), axis=0)
  # Flag for second file
  print(datetime.now(), "Loaded data from second pickle file")

elif (filetype == "h5py"):
  
  ### for input of turbulent data from JHTDB (h5py format)
  f = h5py.File(filename, 'r')
  keys = list(f.keys())
  array = np.ndarray(shape=(len(keys) - 3,160,128,256,3))
  i = 0 
  for x in keys[:-3:]:
    dset_temp = f[x]
    array[i,:,:,:,:] = dset_temp
    i += 1 
  uvw3D_field = np.swapaxes(array, 1, 3)     
  print(datetime.now(), "Loaded data from h5py file")
  
else:
  print(datetime.now(), "filetype???")

  
# Flag for dataset shape
print(datetime.now(), "Shape of 3D Dataset", uvw3D_field.shape)

# Define empty 2D dataset
uvw2D_sec = np.empty([snapshots,nx,ny,15])

slice_pos = np.array([(nz-1)*0.1,(nz-1)*0.3,(nz-1)*0.5,(nz-1)*0.7,(nz-1)*0.9])
slice_pos = slice_pos.astype(int)

# Ex. 5 cross-sections are used to input of the model
uvw2D_sec[:,:,:,0:3]=uvw3D_field[:,:,:,slice_pos[0],:]
uvw2D_sec[:,:,:,3:6]=uvw3D_field[:,:,:,slice_pos[1],:]
uvw2D_sec[:,:,:,6:9]=uvw3D_field[:,:,:,slice_pos[2],:]
uvw2D_sec[:,:,:,9:12]=uvw3D_field[:,:,:,slice_pos[3],:]
uvw2D_sec[:,:,:,12:]=uvw3D_field[:,:,:,slice_pos[4],:]

print(datetime.now(), "Shape of 2D dataset", uvw2D_sec.shape)

# The data is divide to training/validation data
if not Laptop:
  X_train,X_test,y_train,y_test = train_test_split(uvw2D_sec,uvw3D_field,test_size=0.3,random_state=None)  ###doesn't seam to work, using validation split instead.
  print(datetime.now(), "Shape of training dataset", X_train.shape)

# Input variables
input_field = tf.keras.layers.Input(shape=(nx,ny,15))

# Network structure
x = tf.keras.layers.Conv2D(32, (3,3),activation=act, padding='same')(input_field)
x = tf.keras.layers.Conv2D(32, (3,3),activation=act, padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(16, (3,3),activation=act, padding='same')(x)
x = tf.keras.layers.Conv2D(int(nz/8), (3,3),activation=act, padding='same')(x)
x = tf.keras.layers.Reshape([int(nx/2),int(ny/2),int(nz/8),1])(x)
x = tf.keras.layers.Conv3D(16, (3,3,3),activation=act, padding='same')(x)
x = tf.keras.layers.Conv3D(16, (3,3,3),activation=act, padding='same')(x)
x = tf.keras.layers.UpSampling3D((2,2,8))(x)
x = tf.keras.layers.Conv3D(32, (3,3,3),activation=act, padding='same')(x)
x = tf.keras.layers.Conv3D(32, (3,3,3),activation=act, padding='same')(x)
x_final = tf.keras.layers.Conv3D(3,(3,3,3),activation='linear', padding='same')(x)
# -------- #
model = tf.keras.models.Model(input_field,x_final)
model.compile(optimizer='adam',loss='mse')

# Flag for compiling model
print(datetime.now(), "NN Model compiled")

##################

model_cb = tf.keras.callbacks.ModelCheckpoint('/home/ap2021/rds/hpc-work/Test_Model_Checkpoint.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100,verbose=1)
cb = [model_cb, early_cb]
# Use reduced epochs for first run - original is 5000
if Laptop:
  history = model.fit(uvw2D_sec,uvw3D_field,epochs=EPOCHS,batch_size=4,verbose=1,callbacks=cb,shuffle=True,validation_split=0.2)
else:
  history = model.fit(X_train,y_train,epochs=EPOCHS,batch_size=4,verbose=1,callbacks=cb,shuffle=True,validation_data=[X_test, y_test])
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='/home/ap2021/rds/hpc-work/Test_Model_Results.csv',index=False)

model.save("/home/ap2021/rds/hpc-work/Test_Model")

# Flag for model.save
print(datetime.now(), "Successfully saved trained model")
