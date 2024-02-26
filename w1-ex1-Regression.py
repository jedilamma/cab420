# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 08:28:36 2024

@author: jeff
"""

import pandas
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge

data = pandas.read_csv('../cab420/cab420_examples/CorridorTravelTimes_Merged.csv')
pm_data = data.iloc[:, 20:]
pm_data.head()

mu = pm_data.mean()
sigma = pm_data.std()
pm_data_standarsised = (pm_data - mu) / sigma
X = pm_data_standarsised.iloc[:, 0:-1]
Y = pm_data_standarsised.iloc[:, -1]
X = sm.add_constant(X)

train_samples = int(len(X)*0.6)
val_samples = int(len(X)*0.2)
X_train = X.iloc[:train_samples, :]
Y_train = Y.iloc[:train_samples]
X_val = X.iloc[train_samples:(train_samples + val_samples), :]
Y_val = Y.iloc[train_samples:(train_samples + val_samples)]
X_test = X.iloc[(train_samples + val_samples):, :]
Y_test = Y.iloc[(train_samples + val_samples):]

linear = LinearRegression(fit_intercept = False).fit(X = X_train.to_numpy(), y = Y_train.to_numpy())
fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
ax.bar(range(len(linear.coef_)), linear.coef_)
ax.set_title('Coefficients')
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(X_train.values), label='Predicted')
ax.plot(Y_train, label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(X_val.values), label='Predicted')
ax.plot(Y_val.to_numpy(), label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(X_test.values), label='Predicted')
ax.plot(Y_test.to_numpy(), label='Actual')
ax.set_title('Testing Data')
ax.legend();

#------------------------------------------------------------------------------
# Lasso
#------------------------------------------------------------------------------
lasso_model_1 = Lasso(fit_intercept=False, alpha=0.01).fit(X = X_train.to_numpy(), y = Y_train.to_numpy())
lasso_model_2 = Lasso(fit_intercept=False, alpha=0.1).fit(X = X_train.to_numpy(), y = Y_train.to_numpy())
lasso_model_3 = Lasso(fit_intercept=False, alpha=0.5).fit(X = X_train.to_numpy(), y = Y_train.to_numpy())

fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
w = 0.2
pos = numpy.arange(0, len(linear.coef_), 1)
ax.bar(pos - w*2, linear.coef_, width=w, label='linear')
ax.bar(pos - w, lasso_model_1.coef_, width=w, label='lambda=0.01')
ax.bar(pos, lasso_model_2.coef_, width=w, label='lambda=0.1')
ax.bar(pos + w, lasso_model_3.coef_, width=w, label='lambda=0.5')
ax.legend()
ax.set_title('Coefficients')
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(X_train.values), label='linear')
ax.plot(lasso_model_1.predict(X_train.values), label='lambda=0.01')
ax.plot(lasso_model_2.predict(X_train.values), label='lambda=0.1')
ax.plot(lasso_model_3.predict(X_train.values), label='lambda=0.5')
ax.plot(Y_train, label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(X_val.values), label='linear')
ax.plot(lasso_model_1.predict(X_val.values), label='lambda=0.01')
ax.plot(lasso_model_2.predict(X_val.values), label='lambda=0.1')
ax.plot(lasso_model_3.predict(X_val.values), label='lambda=0.5')
ax.plot(Y_val.to_numpy(), label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(X_test.values), label='linear')
ax.plot(lasso_model_1.predict(X_test.values), label='lambda=0.01')
ax.plot(lasso_model_2.predict(X_test.values), label='lambda=0.1')
ax.plot(lasso_model_3.predict(X_test.values), label='lambda=0.5')
ax.plot(Y_test.to_numpy(), label='Actual')
ax.set_title('Testing Data')
ax.legend();


#------------------------------------------------------------------------------
# Ridge Regression
#------------------------------------------------------------------------------

glenn_1 = Ridge(fit_intercept=False, alpha=0.01).fit(X = X_train.to_numpy(), y = Y_train.to_numpy())
glenn_2 = Ridge(fit_intercept=False, alpha=2.5).fit(X = X_train.to_numpy(), y = Y_train.to_numpy())
glenn_3 = Ridge(fit_intercept=False, alpha=10).fit(X = X_train.to_numpy(), y = Y_train.to_numpy())

fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
w = 0.2
pos = numpy.arange(0, len(linear.coef_), 1)
ax.bar(pos - w*2, linear.coef_, width=w, label='linear')
ax.bar(pos - w, glenn_1.coef_, width=w, label='alpha=0.01')
ax.bar(pos, glenn_2.coef_, width=w, label='alpha=2.5')
ax.bar(pos + w, glenn_3.coef_, width=w, label='alpha=10')
ax.legend()
ax.set_title('Coefficients')
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(X_train.values), label='linear')
ax.plot(glenn_1.predict(X_train.values), label='lambda=0.01')
ax.plot(glenn_2.predict(X_train.values), label='lambda=2.5')
ax.plot(glenn_3.predict(X_train.values), label='lambda=10')
ax.plot(Y_train, label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(X_val.values), label='linear')
ax.plot(glenn_1.predict(X_val.values), label='lambda=0.01')
ax.plot(glenn_2.predict(X_val.values), label='lambda=2.5')
ax.plot(glenn_3.predict(X_val.values), label='lambda=10')
ax.plot(Y_val.to_numpy(), label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(X_test.values), label='linear')
ax.plot(glenn_1.predict(X_test.values), label='lambda=0.01')
ax.plot(glenn_2.predict(X_test.values), label='lambda=2.5')
ax.plot(glenn_3.predict(X_test.values), label='lambda=10')
ax.plot(Y_test.to_numpy(), label='Actual')
ax.set_title('Testing Data')
ax.legend();