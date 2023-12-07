
# plot fig2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

FDMA = [4.206, 4.397, 4.564, 4.690, 4.765, 4.797, 4.840, 4.830, 4.814]
CA = [4.754, 4.951, 5.139, 5.289, 5.367, 5.428, 5.461, 5.459, 5.447]
STBC = [4.919, 5.119, 5.306, 5.455, 5.536, 5.591, 5.632, 5.630, 5.621]

font1 = FontProperties(fname='times.ttf', size=27)
font2 = FontProperties(fname='times.ttf', size=23)
font3 = FontProperties(fname='times.ttf', size=15)

fig,ax = plt.subplots(figsize=(10,8))
plt.xlabel('Training ratio', fontproperties=font1)
plt.ylabel('Spectral efficiency (bit/s/Hz)', fontproperties=font1)
index = np.arange(len(STBC)) + 1
plt.xticks(fontproperties=font2)
plt.yticks(fontproperties=font2)
plt.xticks(index, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
plt.ylim(4.0,5.8)

plt.plot(index, FDMA, linewidth = 2, marker = '^', markersize=10, 
         color = '#C4D7A0', linestyle = '-', label='FDMA')
plt.plot(index, CA, linewidth = 2, marker = 's', markersize=10, 
         color = '#FEDA65', linestyle = '-', label='CA without STBC')
plt.plot(index, STBC, linewidth = 2, marker = 'o', markersize=10, 
         color = '#D9958F', linestyle = '-', label='CA with STBC')

plt.legend(loc = 'lower right', prop=font3)
ax.grid(True, ls=':', color='black', alpha=0.3)
plt.savefig('picture/fig2.jpg')
plt.show()
