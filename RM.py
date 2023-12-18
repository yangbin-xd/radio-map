
# radio map method
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
from keras.models import load_model
from utils import *

# Read Data
BS = np.load('data/BSloc.npy')
UEloc = np.load('data/UEloc.npy')
CSI1 = np.load('data/CSI1.npy')
CSI2 = np.load('data/CSI2.npy')

# preprocess
BS1 = np.round(BS[0,0:2], 3)
BS2 = np.round(BS[1,0:2], 3)
loc = UEloc[:,0:2]
CSI1 = np.transpose(CSI1, [0,3,1,2])
CSI2 = np.transpose(CSI2, [0,3,1,2])
[N, Nc, Nr, Nt] = (CSI1.shape)


# CSI concatenate
CSI1_conc = np.empty([N, Nc, Nr, Nt, 2])
CSI2_conc = np.empty([N, Nc, Nr, Nt, 2])
CSI1_conc[:,:,:,:,0] = np.real(CSI1)
CSI1_conc[:,:,:,:,1] = np.imag(CSI1)
CSI2_conc[:,:,:,:,0] = np.real(CSI2)
CSI2_conc[:,:,:,:,1] = np.imag(CSI2)

# input normalization
loc_norm = input_norm(loc)
CSI1_norm = CSI1_conc/np.max(np.abs(CSI1_conc))
CSI1_norm = CSI1_norm[:,:,:,:,0] + 1j * CSI1_norm[:,:,:,:,1]
CSI2_norm = CSI2_conc/np.max(np.abs(CSI2_conc))
CSI2_norm = CSI2_norm[:,:,:,:,0] + 1j * CSI2_norm[:,:,:,:,1]

# shuffle
np.random.seed(1)
np.random.shuffle(loc_norm) # (N, 2)
np.random.seed(1)
np.random.shuffle(CSI1) # (N, Nc, Nr, Nt)
np.random.seed(1)
np.random.shuffle(CSI2) # (N, Nc, Nr, Nt)
np.random.seed(1)
np.random.shuffle(CSI1_norm) # (N, Nc, Nr, Nt)
np.random.seed(1)
np.random.shuffle(CSI2_norm) # (N, Nc, Nr, Nt)

# training ratio
a = 0.5
N_train = int(N * a)
N_test = N - N_train
x_train = loc_norm[0:N_train,:]
y1_train = CSI1_norm[0:N_train,:]
y2_train = CSI2_norm[0:N_train,:]
x_test = loc_norm[N-N_test:N,:]
y1_test = CSI1_norm[N-N_test:N,:]
y2_test = CSI2_norm[N-N_test:N,:]

# # train
# model1 = train(x_train, y1_train, x_test, y1_test, 2000)
# model1.save('model/BS1.h5')
# model2 = train(x_train, y2_train, x_test, y2_test, 2000)
# model2.save('model/BS2.h5')

CSI1_test = CSI1[N-N_test:N,:] # (N_test, Nc, Nr, Nt)
CSI2_test = CSI2[N-N_test:N,:] # (N_test, Nc, Nr, Nt)
model1 = load_model('model/BS1.h5', 
                    custom_objects={'cust_loss': cust_loss})
model2 = load_model('model/BS2.h5', 
                    custom_objects={'cust_loss': cust_loss})

# calculate transmit precoding vector
v1 = model1.predict(x_test)
v1 = v1.reshape(-1, Nc, Nt, 2)
v1 = v1[:,:,:,0] + 1j * v1[:,:,:,1]
v1 = v1.reshape(-1, Nc, Nt, 1)
v1 = output_norm(v1)

v2 = model2.predict(x_test)
v2 = v2.reshape(-1, Nc, Nt, 2)
v2 = v2[:,:,:,0] + 1j * v2[:,:,:,1]
v2 = v2.reshape(-1, Nc, Nt, 1)
v2 = output_norm(v2)

# calculate SE of LoS and NLoS conditions
rate_Nc = STBC(CSI1_test, v1, CSI2_test, v2)
LoS_SE, NLoS_SE = compute(rate_Nc, N_test)
print('Ratio:', a)
print('RM LoS:', np.round(LoS_SE,3), 'bits/s/Hz')
print('RM NLoS:', np.round(NLoS_SE,3), 'bits/s/Hz')

rate_STBC = np.mean(rate_Nc)
print('CA with STBC:', np.round(rate_STBC,3), 'bits/s/Hz')

rate_Nc = CA(CSI1_test, v1, CSI2_test, v2)
rate_CA = np.mean(rate_Nc)
print('CA without STBC:', np.round(rate_CA,3), 'bits/s/Hz')

# FDMA
v1_FDMA = v1
v2_FDMA = v2
v1_FDMA[:,int(Nc/2):Nc,:,:]=0
v2_FDMA[:,0:int(Nc/2),:,:]=0

v1_FDMA = output_norm(v1_FDMA)
v2_FDMA = output_norm(v2_FDMA)

rate_Nc = FDMA(CSI1_test, v1_FDMA, CSI2_test, v2_FDMA)
rate_FDMA = np.mean(rate_Nc)
print('FDMA:', np.round(rate_FDMA,3), 'bits/s/Hz')

