# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:33:08 2024

@author: jeff
"""

import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn

import statsmodels.api
import statsmodels.formula.api as sm
import scipy.stats as stats

### data pre processing

data = pandas.read_csv('../cab420/data/washintonbikeshare/day.csv')
data.head()

fig = plt.figure(figsize=[10,6])
ax=fig.add_subplot(1,1,1)
ax.plot(data['cnt'])
ax.set_xlabel('Observation')
ax.set_ylabel('Number of lycra enthusiasts')
ax.set_title('Total Cyclists over Time')


data_train = data.loc[(data['yr']==0)&(data['mnth']<10)]
data_test = data.loc[(data['yr']==0)&(data['mnth']>=10)]

print(data_train.shape)
print(data_test.shape)

fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(2, 2, 1)
ax.scatter(data_train['atemp'], data_train['cnt'])
ax.set_xlabel('atemp')
ax.set_ylabel('Number of People in Lyrca')
ax.set_title(str(data_train['cnt'].corr(data_train['atemp'])))

ax = fig.add_subplot(2, 2, 2)
ax.scatter(data_train['temp'], data_train['cnt'])
ax.set_xlabel('temp')
ax.set_ylabel('Number of People in Lyrca')
ax.set_title(str(data_train['cnt'].corr(data_train['temp'])))

ax = fig.add_subplot(2, 2, 3)
ax.scatter(data_train['hum'], data_train['cnt'])
ax.set_xlabel('hum')
ax.set_ylabel('Number of People in Lyrca')
ax.set_title(str(data_train['cnt'].corr(data_train['hum'])))

ax = fig.add_subplot(2, 2, 4)
ax.scatter(data_train['windspeed'], data_train['cnt'])
ax.set_xlabel('windspeed')
ax.set_ylabel('Number of People in Lyrca')
ax.set_title(str(data_train['cnt'].corr(data_train['windspeed'])))

fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(1, 1, 1)
ax.scatter(data_train['atemp'], data_train['temp'])
ax.set_xlabel('atemp')
ax.set_ylabel('temp')
ax.set_title(str(data_train['atemp'].corr(data_train['temp'])))



##-------------------------------------------------------------------------------
# part 2 - A Simple Model
##-------------------------------------------------------------------------------

model = sm.ols(formula = "cnt~ atemp + temp +hum + windspeed",data=data_train).fit()
print(model.params)
print(model.summary())

f = statsmodels.api.qqplot(model.resid)

fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(1, 1, 1)
ax.hist(model.resid, 20);

fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(1, 1, 1)
ax.scatter(model.predict(data_train), model.resid);
ax.set_xlabel('Prediction')
ax.set_ylabel('Residual')

print(model.summary())

fig = plt.figure(figsize=[24, 8])
ax = fig.add_subplot(1, 3, 1)
ax.scatter(data_train['atemp'], data_train['cnt'])
ax.set_xlabel('atemp')
ax.set_ylabel('cnt')
ax.set_title(str(data_train['atemp'].corr(data_train['cnt'])))
ax = fig.add_subplot(1, 3, 2)
ax.scatter(data_train['temp'], data_train['cnt'])
ax.set_xlabel('temp')
ax.set_ylabel('cnt')
ax.set_title(str(data_train['temp'].corr(data_train['cnt'])))
ax = fig.add_subplot(1, 3, 3)
ax.scatter(data_train['atemp'], data_train['temp'])
ax.set_xlabel('atemp')
ax.set_ylabel('temp')
ax.set_title(str(data_train['atemp'].corr(data_train['temp'])))


fig = plt.figure(figsize=[6, 6])
ax = fig.add_subplot(1, 1, 1)
# concat X and y so that we get correlations between pairs of predictors, and correlations between 
# predictors and the response. As we've put these in the order [x, y], our correlations with the
# response will be the last column and row
seaborn.heatmap(data_train[["atemp", "temp", "hum", "windspeed", "cnt"]].corr(), ax=ax, vmin=-1.0, vmax=1.0)



def plot_predictions(model, train, test):
    pred_train = model.predict(train)
    pred_test = model.predict(test)
    rmse_train = numpy.sqrt(numpy.mean((pred_train - train['cnt'])**2))
    rmse_test = numpy.sqrt(numpy.mean((pred_test - test['cnt'])**2))

    fig = plt.figure(figsize=[12, 6])
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(train['cnt'], pred_train)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predictions')
    ax.set_title('Training Data (RMSE %f)' % rmse_train)

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(test['cnt'], pred_test)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predictions')
    ax.set_title('Testing Data (RMSE %f)' % rmse_test)
    
plot_predictions(model, data_train, data_test)

##-------------------------------------------------------------------------------
#       Improving the model
##-------------------------------------------------------------------------------


model_2 = sm.ols(formula="cnt ~ temp + hum + windspeed", data=data_train).fit()
print(model_2.summary())

def plot_diagnostics(model, training_data):
    fig = plt.figure(figsize=[24, 8])
    ax = fig.add_subplot(1, 3, 1)    
    f = statsmodels.api.qqplot(model.resid, ax=ax)

    ax = fig.add_subplot(1, 3, 2)
    ax.hist(model.resid, 20)
    
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(model.predict(training_data), model.resid)
    
plot_diagnostics(model_2, data_train)

pred_train_orig = model.predict(data_train)
pred_test_orig = model.predict(data_test)
pred_train = model_2.predict(data_train)
pred_test = model_2.predict(data_test)
fig = plt.figure(figsize=[12, 8])
ax = fig.add_subplot(2, 1, 1)
ax.plot(pred_train_orig, label='Predicted (Model 1)')
ax.plot(pred_train, label='Predicted (Model 2)')
ax.plot(data_train['cnt'], label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(2, 1, 2)
ax.plot(pred_test_orig, label='Predicted (Model 1)')
ax.plot(pred_test, label='Predicted (Model 2)')
ax.plot(data_test['cnt'], label='Actual')
ax.set_title('Testing Data')
ax.legend()

plot_predictions(model, data_train, data_test)


import numpy
rmse_model1_train = numpy.sqrt(numpy.mean((pred_train_orig - data_train['cnt'])**2))
rmse_model1_test = numpy.sqrt(numpy.mean((pred_test_orig - data_test['cnt'])**2))
rmse_model2_train = numpy.sqrt(numpy.mean((pred_train - data_train['cnt'])**2))
rmse_model2_test = numpy.sqrt(numpy.mean((pred_test - data_test['cnt'])**2))

print('Model 1, Training Data RMSE: %f\nModel 1, Testing Data RMSE: %f\nModel 2, Training Data RMSE: %f\nModel 2, Testing Data RMSE: %f' % (rmse_model1_train, rmse_model1_test, rmse_model2_train, rmse_model2_test))

##-------------------------------------------------------------------------------
#       Randomly split data
##-------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data.loc[(data['yr'] == 0)], test_size=0.25, random_state=4)

model_3 = sm.ols(formula="cnt ~ temp + hum + windspeed", data=data_train).fit()
print(model_3.summary())
    
plot_diagnostics(model_3, data_train)
plot_predictions(model_3, data_train, data_test)


##-------------------------------------------------------------------------------
#      Making things more complex
##-------------------------------------------------------------------------------

#Categorical Variables
data_train["weekday"] = data_train["weekday"].astype("category")
data_train["season"] = data_train["season"].astype("category")

data_test["weekday"] = data_test["weekday"].astype("category")
data_test["season"] = data_test["season"].astype("category")

model_4 = sm.ols(formula="cnt ~ temp + hum + windspeed + weekday + season", data=data_train).fit()
print(model_4.summary())

model_5 = sm.ols(formula="cnt ~ temp + hum + windspeed + season", data=data_train).fit()
print(model_5.summary())

plot_diagnostics(model_5, data_train)
plot_predictions(model_5, data_train, data_test)


model_6 = sm.ols(formula="cnt ~ temp * hum * windspeed * season", data=data_train).fit()
print(model_6.summary())

plot_diagnostics(model_6, data_train)
plot_predictions(model_6, data_train, data_test)


model_6 = sm.ols(formula="cnt ~ temp * hum * windspeed * season * numpy.power(temp, 2) * numpy.power(hum, 2) * numpy.power(windspeed, 2)", data=data_train).fit()
print(model_6.summary())

plot_diagnostics(model_6, data_train)
plot_predictions(model_6, data_train, data_test)
