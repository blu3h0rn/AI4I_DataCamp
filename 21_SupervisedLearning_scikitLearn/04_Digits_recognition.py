# In the following exercises, you'll be working with the MNIST digits recognition dataset,
# which has 10 classes, the digits 0 through 9! A reduced version of the MNIST dataset is
# one of scikit-learn's included datasets, and that is the one we will use in this exercise.

# Each sample in this scikit-learn dataset is an 8x8 image representing a handwritten digit.
# Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black.
# Recall that scikit-learn's built-in datasets are of type Bunch, which are dictionary-like objects.
# Helpfully for the MNIST dataset, scikit-learn provides an 'images' key in addition to the 'data' and
# 'target' keys that you have seen with the Iris data. Because it is a 2D array of the images corresponding
# to each sample, this 'images' key is useful for visualizing the images, as you'll see in this exercise
# (for more on plotting 2D arrays, see Chapter 2 of DataCamp's course on Data Visualization with Python).
# On the other hand, the 'data' key contains the feature array - that is, the images as a flattened array
# of 64 pixels.

# Import necessary modules
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.data.shape)
print(digits.images.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# -------------------------------------------------------------------------------------

# Import necessary modules

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

# ---------------------------Fittting curve--------------------------------------
# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
