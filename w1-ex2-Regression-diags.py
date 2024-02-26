# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:10:25 2024

@author: jeff
"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# get some x values
x = np.linspace(0, np.pi * 0.5, 1000)

# seeding random number generator so get consistent results across consecutive runs
np.random.seed(1)

# true line is y = x, add a small amount of noise to get some data to fit
y1 = x + np.random.randn(*x.shape) * 0.1

# fit OLS model
# note we're not adding a constant here, as our equation is y = x, so the constant is 0
ols = sm.OLS(y1, x)
model = ols.fit()
model.summary()

def plot_fit(x, y, y_pred):
    plt.scatter(y, y_pred, label="Data")
    plt.xlabel('Y (actual)')
    plt.ylabel('Y (predicted)')
    plt.plot(x, y_pred, c='red', label="Fitted Model")
    plt.legend()

plot_fit(x, y1, model.predict(x))


def plot_resid(resid, pred):
    fig = plt.figure(figsize=[20, 6])
    ax = fig.add_subplot(1, 3, 1)
    ax.hist(resid, bins=50)
    ax.set_title('Histogram of Residuals')
    
    ax = fig.add_subplot(1, 3, 2)
    sm.qqplot(np.squeeze(resid), line='s', ax=ax)
    ax.set_title('QQ Plot')

    ax = fig.add_subplot(1, 3, 3)
    plt.scatter(pred, resid)
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Residual')
    ax.set_title('Residual Scatter Plot')
    
plot_resid(model.resid, model.predict(x))

#------------------------------------------------------------------------------------------
# SCENARIO 2 - NON LINEAR DATA
#------------------------------------------------------------------------------------------

# seeding random number generator so get consistent results
np.random.seed(1)
y2 = np.sin(x) + np.random.randn(*x.shape) * 0.02
plt.figure()
plt.scatter(x, y2)

x_c = sm.add_constant(x)
ols = sm.OLS(y2, x_c)
model = ols.fit()
model.summary()

plt.figure()
plot_fit(x, y2, model.predict(x_c))

plot_resid(model.resid, model.predict(x_c))

#------------------------------------------------------------------------------------------
# Scenario 3: Non-constant Variance
#------------------------------------------------------------------------------------------

# seeding random number generator so get consistent results
np.random.seed(1)

a = 1
b = 0

y3 = a * x + b + np.random.randn(*x.shape) * x 

x_c = sm.add_constant(x)
ols = sm.OLS(y3, x_c)
model = ols.fit()
model.summary()

plt.figure()
plot_fit(x, y3, model.predict(x_c))

plt.figure()
plot_resid(model.resid, model.predict(x_c))