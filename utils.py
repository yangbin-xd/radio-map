
# the library of functions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

# read data
CSI = np.load('data/CSI1.npy')
(N, Nr, Nt, Nc) = CSI.shape # (2000, 2, 32, 64)

# input normalization
def input_norm(loc):
    x_min = np.min(loc[:,1])
    x_max = np.max(loc[:,1])
    y_min = np.min(loc[:,0])
    y_max = np.max(loc[:,0])
    x_len = x_max - x_min
    y_len = y_max - y_min
    loc_norm = np.empty(loc.shape)
    loc_norm[:,0] = (loc[:,0] - y_min)/y_len
    loc_norm[:,1] = (loc[:,1] - x_min)/x_len
    return loc_norm

# output normalization
def output_norm(v):
    power = np.matmul(np.transpose(np.conj(v), (0,1,3,2)), v)
    power = np.sum(power.reshape(-1, Nc), axis=-1).reshape(-1, 1)
    power = np.matmul(power, np.ones((1, Nc)))
    power = power.reshape(-1, Nc, 1, 1)
    v_norm = np.sqrt(Nc) * v / np.sqrt(power)
    return v_norm

# calculate SE of STBC
def STBC(H1, v1, H2, v2):
    Hv1 = np.matmul(H1, v1)
    Hv1_gain = np.matmul(np.transpose(np.conj(Hv1), (0,1,3,2)), Hv1)
    Hv1_gain = np.squeeze(np.abs(Hv1_gain))

    Hv2 = np.matmul(H2, v2)
    Hv2_gain = np.matmul(np.transpose(np.conj(Hv2), (0,1,3,2)), Hv2)
    Hv2_gain = np.squeeze(np.abs(Hv2_gain))

    noise = 3.9811e-10 # -94dBm

    SNR = (Hv1_gain + Hv2_gain) / noise
    rate = np.log2(1 + SNR)  # rate
    rate_Nc = np.mean(rate, axis=1)  # average of subcarriers
    return rate_Nc

# calculate SE of CA
def CA(H1, v1, H2, v2):
    Hv1 = np.matmul(H1, v1)
    Hv2 = np.matmul(H2, v2)
    Hv = Hv1 + Hv2
    Hv_gain = np.matmul(np.transpose(np.conj(Hv), (0,1,3,2)), Hv)
    Hv_gain = np.squeeze(np.abs(Hv_gain))

    noise = 3.9811e-10 # -94dBm

    SNR = Hv_gain / noise
    rate = np.log2(1 + SNR)  # rate
    rate_Nc = np.mean(rate, axis=1)  # average of subcarriers
    return rate_Nc

# calculate SE of FDMA
def FDMA(H1, v1, H2, v2):
    Hv1 = np.matmul(H1, v1)
    Hv1_gain = np.matmul(np.transpose(np.conj(Hv1), (0,1,3,2)), Hv1)
    Hv1_gain = np.squeeze(np.abs(Hv1_gain))
    Hv2 = np.matmul(H2, v2)
    Hv2_gain = np.matmul(np.transpose(np.conj(Hv2), (0,1,3,2)), Hv2)
    Hv2_gain = np.squeeze(np.abs(Hv2_gain))

    noise = 3.9811e-10 # -94dBm
    SNR1 = Hv1_gain / noise
    rate1 = np.log2(1 + SNR1)  # rate
    rate1_Nc = np.mean(rate1, axis=1)  # average of subcarriers
    SNR2 = Hv2_gain / noise
    rate2 = np.log2(1 + SNR2)  # rate
    rate2_Nc = np.mean(rate2, axis=1)  # average of subcarriers
    rate_Nc = rate1_Nc + rate2_Nc
    return rate_Nc

# compute the SE of LoS and NLoS conditions
def compute(rate_Nc, N_test):
    LoS = np.load('data/LoS.npy')
    NLoS = np.load('data/NLoS.npy')

    index = np.zeros(N)
    for i in np.arange(LoS.shape[0]):
        index[LoS[i]] = 1
    for i in np.arange(NLoS.shape[0]):
        index[NLoS[i]] = -1

    np.random.seed(1)
    np.random.shuffle(index)
    index = index[2000 - N_test:2000]

    LoS_num = 0
    NLoS_num = 0
    LoS_SE = 0
    NLoS_SE = 0
    for i in np.arange(index.shape[0]):
        if index[i] == 1:
            LoS_SE = LoS_SE + rate_Nc[i]
            LoS_num = LoS_num + 1
        if index[i] == -1:
            NLoS_SE = NLoS_SE + rate_Nc[i]
            NLoS_num = NLoS_num + 1

    LoS_SE = LoS_SE / LoS_num
    NLoS_SE = NLoS_SE / NLoS_num

    return LoS_SE, NLoS_SE

# customized loss function
def cust_loss(H, v):
    v = tf.reshape(v, [-1, Nc, Nt, 2])
    v_comp = tf.complex(v[:,:,:,0], v[:,:,:,1])
    v_conj = tf.complex(v[:,:,:,0], -v[:,:,:,1])
    v_comp = tf.reshape(v_comp, [-1, Nc, Nt, 1])
    v_conj = tf.reshape(v_conj, [-1, Nc, Nt, 1])

    power = tf.matmul(tf.transpose(v_conj, (0,1,3,2)), v_comp)
    power = tf.reduce_sum(tf.reshape(power, [-1, Nc]), axis=-1)
    power = tf.reshape(power, [-1, 1])
    power = tf.matmul(power, tf.dtypes.cast(tf.ones((1, Nc)), tf.complex64))
    power = tf.reshape(power, [-1, Nc, 1, 1])
    v = tf.sqrt(tf.dtypes.cast(Nc, tf.complex64)) * v_comp / tf.sqrt(power)
    
    Hv = tf.matmul(tf.dtypes.cast(H, tf.complex64), v)
    Hv_conj = tf.transpose(Hv, (0,1,3,2), conjugate=True)
    Hv_gain = tf.matmul(Hv_conj, Hv)
    Hv_gain = tf.reshape(tf.abs(Hv_gain), [-1, Nc])

    noise = 3.9811e-10
    SNR = Hv_gain / noise
    rate = tf.math.log(1 + SNR)/np.log(2)  # rate
    rate_Nc = tf.reduce_mean(rate, axis=0)  # average of subcarriers
    rate_mean = tf.reduce_mean(rate_Nc)  # average of N_test
    loss = -rate_mean
    return loss

# train
def train(x_train, y_train, x_test, y_test, epoch):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, input_dim=2, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(1000, activation='relu'),
            tf.keras.layers.Dense(Nc * Nt * 2, activation='tanh')])
    model.summary()
    model.compile(loss = cust_loss, optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4))
    model.fit(x_train, y_train, epochs = epoch, batch_size = 128, verbose = 1)
    model.evaluate(x_test, y_test, verbose = 1)
    return model
  