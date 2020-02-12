# import all needed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# read in boston as dataframe
boston = pd.read_csv(
    '/Users/apple/Documents/Online_Courses/DataCamp_Excercise/21_SupervisedLearning_scikitLearn/boston.csv')

print(boston.head())

# extract feature and target - becomes numpy arrays
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values

# predict house value based on 1 feature - average number of rooms
# slice the room number from dataframe x
X_rooms = X[:, 5]
print(X_rooms)
# check for datatypes
print(type(boston), type(X), type(X_rooms), type(y))

# to add another dimension to X_rooms and y, do reshape
# -1 means unknown row, 1 is to number of column
y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

# plot house price vs number of rooms
plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
# plt.show()

# fit linear regression to the plot
# create the regressor
reg = LinearRegression()
#  fit in no of rooms and y
reg.fit(X_rooms, y)
# create the prediction space
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)

# plot house price vs number of rooms
plt.scatter(X_rooms, y, color='blue')

# compute the predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# compute the R2 score
print(reg.score(X_rooms, y))  # --> 0.483525

# plot the prediction line over the y_pred
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()
