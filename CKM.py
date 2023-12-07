
# channel knowledge map
import numpy as np
from library import *

# Read Data
UEloc = np.load('data/UEloc.npy')
CSI1 = np.load('data/CSI1.npy')
CSI2 = np.load('data/CSI2.npy')
DoD1 = np.load('data/DoD1.npy')
DoD2 = np.load('data/DoD2.npy')

# preprocess
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
np.random.seed(1)
np.random.shuffle(DoD1) # (N, )
np.random.seed(1)
np.random.shuffle(DoD2) # (N, )

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
DoD1_train = DoD1[0:N_train]
DoD1_test = DoD1[N-N_test:N]
DoD2_train = DoD2[0:N_train]
DoD2_test = DoD2[N-N_test:N]

# calculate distance to apply inverse distance weighting (IDW)
dist = np.empty([N_test, N_train])
for i in range(N_test):
    for j in range(N_train):
        dist[i,j] = np.sqrt(np.sum((loc_test[i] - loc_train[j]) ** 2))

# K nearest neighbors
def KNN(i, DoD_train, k):
    dist_sort = sorted(enumerate(dist[i,:]), key=lambda x:x[1])
    index = [x[0] for x in dist_sort]
    K = index[0:k]
    dist_k = dist[i,K]
    DoD_k = np.squeeze(DoD_train[K])
    weight = 1/dist_k
    weight = weight/np.sum(weight)
    DoD_IDW = np.dot(weight, DoD_k)
    return DoD_IDW

# communication parameters
pi = np.pi
c = 3e8
fc = 3.5e9
lamda = c/fc
B = 100e6
d = 1/2*lamda

# calculate transmit precoding vector
v1 = np.empty([N_test, Nc, Nt, 1], dtype=complex)
v2 = np.empty([N_test, Nc, Nt, 1], dtype=complex)
for i in range(N_test):
    theta1 = KNN(i, DoD1_train, 3) * pi / 180
    theta2 = KNN(i, DoD2_train, 3) * pi / 180
    for j in range(Nc):
        lamda = c / (fc + j*B/Nc)
        for n in range(Nt):
            v1[i,j,n] = np.exp(-1j*2*pi*n*d*np.sin(theta1)/lamda)
            v2[i,j,n] = np.exp(-1j*2*pi*n*d*np.sin(theta2)/lamda)

# output normalization
v1 = output_norm(v1)
v2 = output_norm(v2)

# calculate SE of LoS and NLoS conditions
rate_Nc = STBC(CSI1_test, v1, CSI2_test, v2)
LoS_SE, NLoS_SE = compute(rate_Nc, N_test)
print('Ratio:', a)
print('CKM LoS:', np.round(LoS_SE,3), 'bits/s/Hz')
print('CKM NLoS:', np.round(NLoS_SE,3), 'bits/s/Hz')