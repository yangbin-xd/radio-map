
# location based beamforming
import math
import numpy as np
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

# shuffle
np.random.seed(1)
np.random.shuffle(loc) # (N, 2)
np.random.seed(1)
np.random.shuffle(CSI1) # (N, Nc, Nr, Nt)
np.random.seed(1)
np.random.shuffle(CSI2) # (N, Nc, Nr, Nt)

# training ratio
a = 0.5
N_train = int(N * a)
N_test = N - N_train
loc_train = loc[0:N_train,:]
loc_test = loc[N-N_test:N,:]
CSI1_train = CSI1[0:N_train,:]
CSI1_test = CSI1[N-N_test:N,:]
CSI2_train = CSI2[0:N_train,:]
CSI2_test = CSI2[N-N_test:N,:]

# calculate DoD based on locations
DoD1 = np.empty(N_test)
DoD2 = np.empty(N_test)
pi = np.pi
for i in np.arange(N_test):
    DoD1[i] = math.atan((loc_test[i,1] - BS1[1]) / 
                        (loc_test[i,0] - BS1[0])) * 180 / pi
    DoD2[i] = math.atan((loc_test[i,1] - BS2[1]) /
                        (loc_test[i,0] - BS2[0])) * 180 / pi

# communication parameters
c = 3e8
f = 3.5e9
lamda = c/f
B = 100e6
d = 1/2*lamda

# calculate transmit precoding vector
v1 = np.empty([N_test, Nc, Nt, 1], dtype=complex)
v2 = np.empty([N_test, Nc, Nt, 1], dtype=complex)
for i in range(N_test):
    DoD1[i] = DoD1[i] * pi / 180
    DoD2[i] = DoD2[i] * pi / 180
    for j in range(Nc):
        lamda = c / (f + j*B/Nc)
        for n in range(Nt):
            v1[i,j,n] = np.exp(-1j*2*pi*n*d*np.sin(DoD1[i])/lamda)
            v2[i,j,n] = np.exp(-1j*2*pi*n*d*np.sin(DoD2[i])/lamda)

# output normalization
v1 = output_norm(v1)
v2 = output_norm(v2)

# calculate SE of LoS and NLoS conditions
rate_Nc = STBC(CSI1_test, v1, CSI2_test, v2)
LoS_SE, NLoS_SE = compute(rate_Nc, N_test)
print('Ratio:', a)
print('LBB LoS:', np.round(LoS_SE,3), 'bits/s/Hz')
print('LBB NLoS:', np.round(NLoS_SE,3), 'bits/s/Hz')
