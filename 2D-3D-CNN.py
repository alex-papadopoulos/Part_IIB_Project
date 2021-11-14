'''
2D-3D Convolutional Neural Networks
'''

# To run this code, a listed modules are required
from keras.layers import Input,Conv2D, Conv3D, MaxPooling2D, UpSampling3D, Reshape
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.backend import tf as ktf
import pickle
import numpy as np
import pandas as pd


# Parameters
act = 'relu'

# Configurations of input/output variables are as follows:
'''
3D_field: time, nx =256, ny=128, nz=160, components=3
'''

# Define empty 2D dataset
uvw2D_sec = np.empty([50,256,128,15])

# Ex. 5 cross-sections are used to input of the model
uvw2D_sec[:,:,:,0:3]=uvw3D_field[:,:,:,15,:]
uvw2D_sec[:,:,:,3:6]=uvw3D_field[:,:,:,47,:]
uvw2D_sec[:,:,:,6:9]=uvw3D_field[:,:,:,79,:]
uvw2D_sec[:,:,:,9:12]=uvw3D_field[:,:,:,111,:]
uvw2D_sec[:,:,:,12:]=uvw3D_field[:,:,:,143,:]

# The data is divide to training/validation data
X_train,X_test,y_train,y_test = train_test_split(uvw2D_sec,uvw3D_field,test_size=0.3,random_state=None)


# Input variables
input_field = Input(shape=(256,128,15))

# Network structure
x = Conv2D(32, (3,3),activation=act, padding='same')(input_field)
x = Conv2D(32, (3,3),activation=act, padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(16, (3,3),activation=act, padding='same')(x)
x = Conv2D(20, (3,3),activation=act, padding='same')(x)
x = Reshape([128,64,20,1])(x)
x = Conv3D(16, (3,3,3),activation=act, padding='same')(x)
x = Conv3D(16, (3,3,3),activation=act, padding='same')(x)
x = UpSampling3D((2,2,8))(x)
x = Conv3D(32, (3,3,3),activation=act, padding='same')(x)
x = Conv3D(32, (3,3,3),activation=act, padding='same')(x)
x_final = Conv3D(3,(3,3,3),activation='linear', padding='same')(x)
# -------- #
model = Model(input_field,x_final)
model.compile(optimizer='adam',loss='mse')

##################

from keras.callbacks import ModelCheckpoint,EarlyStopping
model_cb=ModelCheckpoint('./Model_cy.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=100,verbose=1)
cb = [model_cb, early_cb]
history = model.fit(X_train,y_train,epochs=5000,batch_size=100,verbose=1,callbacks=cb,shuffle=True,validation_data=[X_test, y_test])
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./Model_cy.csv',index=False)

model.save("./my_model")