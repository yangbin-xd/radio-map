
# plot the scenario of DeepMIMO
import numpy as np
import matplotlib.pyplot as plt

# read data
BS = np.load('data/BSloc.npy')
UEloc = np.load("data/UEloc.npy")
LoS = np.load('data/LoS.npy')
NLoS = np.load('data/NLoS.npy')

# preprocess
BS1 = np.round(BS[0,0:2], 3)
BS2 = np.round(BS[1,0:2], 3)
loc = UEloc[:,0:2]
Block = np.setdiff1d(np.arange(loc.shape[0]), np.union1d(LoS, NLoS))

# show BS and UE location
fig,ax = plt.subplots(figsize=(16,4))
ax.scatter(BS1[1], BS1[0], marker='^', c='red', s = 50, 
           label='BS1')
ax.scatter(BS2[1], BS2[0], marker='^', c='green', s = 50, 
           label='BS3')
ax.scatter(loc[LoS,1], loc[LoS,0], marker='.', c='yellow', s = 20, 
           label='UE LoS ')
ax.scatter(loc[NLoS,1], loc[NLoS,0], marker='.', c='blue', s = 20, 
           label='UE NLoS')
ax.scatter(loc[Block,1], loc[Block,0], marker='.', c='black', s = 20, 
           label='UE Block')
ax.set_aspect(1)
ax.invert_xaxis()
plt.title('Scenario')
plt.legend(loc = 'upper right')
plt.savefig('picture/scenario.jpg')
plt.show()
