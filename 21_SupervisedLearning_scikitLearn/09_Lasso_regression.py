#In this exercise, you will fit a lasso regression to the Gapminder data you have been 
# working with and plot the coefficients. Just as with the Boston data, you will find 
# that the coefficients of some features are shrunk to 0, with only the most important 
# ones remaining.

# Import Lasso
from sklearn.linear_model import Lasso
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame: df
df = pd.read_csv('/Users/apple/Documents/Online_Courses/DataCamp_Excercise/21_SupervisedLearning_scikitLearn/gapminder.csv', sep=",")

# Create feature (all except life) and target (life) arrays
y = df['life'].values
X = df.drop(['life', 'Region'], axis=1).values

df_columns = df.columns.drop(['life', 'Region'])

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()

