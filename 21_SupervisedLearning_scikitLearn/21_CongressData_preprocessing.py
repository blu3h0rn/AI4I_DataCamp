import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

file= "/Users/apple/Documents/Online_Courses/DataCamp_Excercise/21_SupervisedLearning_scikitLearn/house-votes-84.csv"
df = pd.read_csv(file)


# Convert '?' to NaN
df[df == '?'] =np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
# df = df.dropna()

# Print shape of new DataFrame
# print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

