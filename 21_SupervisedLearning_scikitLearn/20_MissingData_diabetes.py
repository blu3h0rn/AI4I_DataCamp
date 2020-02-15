import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/apple/Documents/Online_Courses/DataCamp_Excercise/21_SupervisedLearning_scikitLearn/diabetes.csv')

# print(df.info())
# print(df.head())

X = df.drop('diabetes', axis=1)
y = df['diabetes']

# replace missing values with NAN
df.insulin.replace(0, np.nan, inplace=True)
df.triceps.replace(0, np.nan, inplace=True)
df.bmi.replace(0, np.nan, inplace=True)
# print(df.info())

# drop missing data --NOT recommended
# df = df.dropna()
# print(df.shape)

# imputate missing data
# making an education guess about missing values
# example: using the mean of the non-missing entries

# instantiate imputer - axis = 0 impute along column, axis=1 along rows
imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
logreg = LogisticRegression()
steps = [('imputation', imp), ('logistic_regression', logreg)]

pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(pipeline.score(X_test, y_test))
