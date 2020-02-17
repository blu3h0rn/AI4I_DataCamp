import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('TrainingData.csv', index_col=0)

# Print the summary statistics
print(df.describe())

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create the histogram
plt.hist(df['FTE'].dropna())

# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')

# Display the histogram
plt.show()

#How many columns with dtype object are in the data?
df.dtypes.value_counts()

# create 9 columns of labels in the dataset
LABELS = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']

# check out the type for these labels 
df[LABELS].dtypes

# convert the labels to category types using the .astype() method.
# .astype() only works on a pandas Series. Since you are working with a pandas DataFrame, 
# you'll need to use the .apply() method and provide a lambda function called 
# categorize_label that applies .astype() to each column, x.

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)

# Print the converted dtypes
print(df[LABELS].dtypes)

#Counting unique labels

# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = df[LABELS].apply(pd.Series.nunique)

# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')

# Display the plot
plt.show()