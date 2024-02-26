# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 07:37:16 2024

@author: jeff
"""

# import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# load in the data from the csv file
# here we are loading the data using a URL, but it could also be a path to a local
# CSV file stored on our machine
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# lets get a quick summary of the data
iris.head()

# lets get the shape of the dataframe
print(iris.shape)

column_names = iris.columns.values
print(column_names)

# first lets try and split them into a training set and a testing set
# we will say use 80% for training and 20% to test our pretend model
# find the number of samples and save it into a variable
# is the first index of the shape property
num_samples = iris.shape[0] 


# now find 80% of that and round it
# will also convert it to an integer as we will be using it for an index
# in just a bit
num_train = np.round(num_samples * 0.8).astype(np.int64)
# now find how many we will use for testing
num_test = num_samples - num_train
# now lets use these to separate into an initial training and test set
# where we will take the first 80% for training and the last 20% for test
#
# We can actually index a pandas Dataframe similar to a numpy array using
# the iloc method
initial_train = iris.iloc[0:num_train, :]
# then get the last 20% for test
# so will be from num_train to the end
initial_test = iris.iloc[num_train::, :]

print(f'train shape = {initial_train.shape}')
print(f'test shape = {initial_test.shape}')

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  
# plot the boxplot for the training data in the first subplot
initial_train.boxplot(ax=ax1)
# set the title
ax1.set_title('Training Data')
# same thing for the testing data
initial_test.boxplot(ax=ax2)
ax2.set_title('Testing Data')

# plot the boxplot for the training data in the first subplot
# and group by the species
#
# not going to bother trying the make them subplots here.
# It can be done, and maybe for people who are keen you could play
# around with it :)
initial_train.boxplot(by='species')
# same thing for the testing data
initial_test.boxplot(by='species')

# we can access an individual column within our dataframe
# by using the dot notation, as long as the column name
# doesn't have a space in it
print(f'Unique species in train = {initial_train.species.unique()}')
print(f'Unique species in test = {initial_test.species.unique()}')

# lets create histograms of just the number of different examples in each dataframe
#
# Again, copying a lot of the plotting code used above
# 
# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  
# plot histogram of just the species column/series 
initial_train.species.hist(ax=ax1)
# set the title
ax1.set_title('Training Data')
# same thing for the testing data
initial_test.species.hist(ax=ax2)
ax2.set_title('Testing Data')


from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(iris,
                                     test_size=num_test, # number of testing points
                                     shuffle=True, # shuffle it about
                                     stratify=iris.species) # make sure distribution of species are balanced

# lets plot it again
# Again, copying a lot of the plotting code used above
# 
# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  
# plot histogram of just the species column/series 
train_df.species.hist(ax=ax1)
# set the title
ax1.set_title('Training Data')
# same thing for the testing data
test_df.species.hist(ax=ax2)
ax2.set_title('Testing Data')

# lets create a dictionary that maps from a species to an integer
species_dict = {'setosa':0,
                'versicolor':1,
                'virginica':2}

# now lets create a function that we can apply to our dataframes to convert the species
# string to the corresponding integer
# the function we are making will only need to work on a single row of our dataframe,
# as we will be using the pandas `apply` method to apply it to every row efficiently

def species_to_int(row):
    # replace the row species string with the integer
    row.species = species_dict[row.species]
    # now return the row
    return row

# now apply this function, with axis specified to one so that it
# will apply the function to every row
train_df_cat = train_df.apply(species_to_int, axis=1)
test_df_cat = test_df.apply(species_to_int, axis=1)

print(train_df_cat.head())
print(test_df_cat.head())

# predictor variables is everything except the species column
train_x = train_df_cat.loc[:, train_df_cat.columns!='species']
test_x = test_df_cat.loc[:, test_df_cat.columns!='species']
# now get the response variables, which is just the species column
train_y = train_df_cat.species
test_y = test_df_cat.species

# now we also want to convert them to numpy arrays
train_x = train_x.to_numpy()
test_x = test_x.to_numpy()
train_y = train_y.to_numpy()
test_y = test_y.to_numpy()

# now lets print them to have a look
print(test_x)
print(test_y)

# indexing the original data frame to give me just the setosa samples
setosa = iris[iris.species == 'setosa']
# checking the shape of this 
print(setosa.shape)
# checking to make sure only have the setosa species
print(setosa.species.unique())
setosa.head()

sepal_width_3_6 = iris[iris.sepal_width >= 3.6]
sepal_width_3_6.head()

# can get stats of a column from pandas dataframes as well
print(sepal_width_3_6.sepal_width.max())
print(sepal_width_3_6.sepal_width.mean())
print(sepal_width_3_6.sepal_width.min())

#---------------------------------------------------------------------------------------------------------

finance = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/GOOG?period1=1582781719&period2=1614404119&interval=1d&events=history&includeAdjustedClose=true")
finance.head()

# lets see the ranges of the dates
print(finance.Date.max())
print(finance.Date.min())

print(finance.dtypes)

finance.Date = finance.Date.astype('datetime64[ns]')
print(finance.dtypes)
finance.head()

finance_2021 = finance[finance.Date >= '2021-01-01']
finance_2021.head()

# note when combining boolean operations from pandas dataframes we need to use & instead of `and`
# and `|` instead of `or`
finance_2020_2nd = finance[(finance.Date >= '2020-07-01') & (finance.Date <= '2020-12-31')]
print(finance_2020_2nd.Date.min())
print(finance_2020_2nd.Date.max())
finance_2020_2nd.head()

result = finance.Open-finance.Close
result.head()
print(result.max())
print(result.min())
