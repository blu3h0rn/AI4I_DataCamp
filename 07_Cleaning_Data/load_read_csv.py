# Import pandas
import pandas as pd

# Read the file into a DataFrame: df
df = pd.read_csv('/Users/apple/Documents/Online_Courses/DataCamp_Excercise/Datasets/dob_job_application_filings_subset.csv')

# Print the head and tail of df
print(df.head())
print(df.tail())

# Print the shape of df
print(df.shape)

print(df.columns)

# unique data
print(df.info())
