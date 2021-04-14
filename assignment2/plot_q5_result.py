# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 42

data = np.loadtxt('./q5_result.txt', delimiter=',')
plt.plot(data[:, 0], data[:, 1])
plt.xlabel('discritization')
plt.ylabel('value')
#plt.legend()
plt.show()
