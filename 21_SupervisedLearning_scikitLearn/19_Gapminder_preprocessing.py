import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Read 'gapminder.csv' into a DataFrame: df
file = "/Users/apple/Documents/Online_Courses/DataCamp_Excercise/21_SupervisedLearning_scikitLearn/gapminder.csv"
df = pd.read_csv(file)

# Create a boxplot of life expectancy per region
# df.boxplot('life', 'Region', rot=60)

# Show the plot
# plt.show()

# --------------Creating dummy variables---------------------------
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)


# Print the new columns of df_region
print(df_region.columns)

# ---------------------Regression with categorical features-------------------
# Create arrays for features and target variable
X = df_region.drop('life', axis=1).values
print(X.shape)
y = df_region['life'].values
print(y.shape)

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)
