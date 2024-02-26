# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:21:42 2024

@author: jeff
"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

#------------------------------------------------------------------------------------------
# SCENARIO 2 - NON LINEAR DATA
#------------------------------------------------------------------------------------------

# seeding random number generator so get consistent results
np.random.seed(1)
y2 = np.sin(x) + np.random.randn(*x.shape) * 0.02
plt.scatter(x, y2)


