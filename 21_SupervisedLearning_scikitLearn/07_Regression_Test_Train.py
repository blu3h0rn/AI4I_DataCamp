# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame: df
df = pd.read_csv('/Users/apple/Documents/Online_Courses/DataCamp_Excercise/21_SupervisedLearning_scikitLearn/gapminder.csv')

# Create feature (all except life) and target (life) arrays
y = df['life'].values
X = df.drop(['life', 'Region'], axis=1).values

# Reshape X and y
y = y.reshape(-1, 1)
X = X.reshape(-1, 8)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 using the .score() method on the test set
print("R^2: {}".format(reg_all.score(X_test, y_test)))

#Compute and print the RMSE. To do this, first compute the Mean Squared Error using 
# the mean_squared_error() function with the arguments y_test and y_pred, and then 
# take its square root using np.sqrt().
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))