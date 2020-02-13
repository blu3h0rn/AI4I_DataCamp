# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame: df
df = pd.read_csv('/Users/apple/Documents/Online_Courses/DataCamp_Excercise/21_SupervisedLearning_scikitLearn/gapminder.csv', sep=",")

# Create feature (all except life) and target (life) arrays
y = df['life'].values
X = df.drop(['life', 'Region'], axis=1).values

# Reshape X and y
y = y.reshape(-1, 1)
X = X.reshape(-1, 8)

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

#Compute and print the average cross-validation score. You can use NumPy's mean() 
# function to compute the average.
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


#-----------------Compare 3 fold and 10 fold CV----------------------------

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print("3-fiold CV: ", np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print("10-fold CV: ", np.mean(cvscores_10))