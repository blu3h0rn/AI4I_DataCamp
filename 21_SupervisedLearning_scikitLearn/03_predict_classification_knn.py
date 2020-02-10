# k-Nearest Neighbors: Predict
# Having fit a k-NN classifier, you can now use it to predict the label of a new data point.
# However, there is no unlabeled data available since all of it was used to fit the model

# You can still use the .predict() method on the X that was used to fit the model,
# but it is not a good indicator of the model's ability to generalize to new, unseen data.

# Import KNeighborsClassifier from sklearn.neighbors
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("house-votes-84.csv")

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
X_new = df.sample()  # random select
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))
