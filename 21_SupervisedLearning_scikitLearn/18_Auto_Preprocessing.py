import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df= pd.read_csv("/Users/apple/Documents/Online_Courses/DataCamp_Excercise/21_SupervisedLearning_scikitLearn/auto.csv")

df_origin = pd.get_dummies(df)
print(df_origin.head())

# drop origin Asia column since already know that if car is not from europe or us, 
# must be from asia
df_origin = df_origin.drop('origin_Asia', axis=1)
print(df_origin.head())



# X_train, X_test, y_train, y_test = train_test_split(X, y)