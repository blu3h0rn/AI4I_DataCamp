import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# Import PCA
from sklearn.decomposition import PCA

# You are given an array grains giving the width and length of samples of grain.
# You suspect that width and length will be correlated.
# To confirm this, make a scatter plot of width vs length and measure their Pearson correlation

df = pd.read_csv(
    "/Users/apple/Documents/Online_Courses/DataCamp_Excercise/23_Unsupervised_Learning/seeds-width-vs-length.csv", header=None)

grains = df.to_numpy()

# Assign the 0th column of grains: width
width = grains[:, 0]

# Assign the 1st column of grains: length
length = grains[:, 1]


# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Decorrelating the grain measurements with PCA

# Create an instance of PCA called model
model = PCA()

# Use the .fit_transform() method of model to apply the PCA transformation to grains. 
# Assign the result to pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)