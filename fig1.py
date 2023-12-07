
# plot fig 1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

LBB_LoS = [5.917, 5.904, 5.893, 5.904, 5.864, 5.864, 5.842, 5.856, 5.870]
LBB_NLoS = [0.377, 0.390, 0.399, 0.413, 0.425, 0.427, 0.465, 0.461, 0.378]

CKM_LoS = [4.910, 5.316, 5.454, 5.536, 5.590, 5.604, 5.676, 5.766, 5.691]
CKM_NLoS = [2.026, 2.484, 2.650, 2.714, 2.838, 2.935, 2.948, 3.039, 3.039]

RM_LoS = [5.283, 5.497, 5.723, 5.908, 6.006, 6.056, 6.087, 6.091, 6.157]
RM_NLoS = [3.432, 3.599, 3.560, 3.661, 3.628, 3.649, 3.670, 3.633, 3.602]

font1 = FontProperties(fname='times.ttf', size=28)
font2 = FontProperties(fname='times.ttf', size=24)
font3 = FontProperties(fname='times.ttf', size=15)

fig,ax = plt.subplots(figsize=(10,8))
plt.xlabel('Training ratio', fontproperties=font1)
plt.ylabel('Spectral efficiency (bit/s/Hz)', fontproperties=font1)
index = np.arange(len(LBB_LoS)) + 1
plt.xticks(fontproperties=font2)
plt.yticks(fontproperties=font2)
plt.xticks(index, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
plt.ylim(0,7)

plt.plot(index, LBB_LoS, linewidth = 2, marker = '^', markersize=10, 
         color = '#C4D7A0', linestyle = '-', label='LBB [9]-[11] LoS')
plt.plot(index, CKM_LoS, linewidth = 2, marker = 's', markersize=10, 
         color = '#FEDA65', linestyle = '-', label='CKM [5]-[7] LoS')
plt.plot(index, RM_LoS, linewidth = 2, marker = 'o', markersize=10, 
         color = '#D9958F', linestyle = '-', label='Radio Map LoS')

plt.plot(index, LBB_NLoS, linewidth = 2, marker = '^', markersize=10, 
         color = '#C4D7A0', linestyle = '--', label='LBB [9]-[11] NLoS')
plt.plot(index, CKM_NLoS, linewidth = 2, marker = 's', markersize=10, 
         color = '#FEDA65', linestyle = '--', label='CKM [5]-[7] NLoS')
plt.plot(index, RM_NLoS, linewidth = 2, marker = 'o', markersize=10, 
         color = '#D9958F', linestyle = '--', label='Radio Map NLoS')

plt.legend(loc = (0.66, 0.09), prop=font3)
ax.grid(True, ls=':', color='black', alpha=0.3)
plt.savefig('picture/fig1.jpg')
plt.show()