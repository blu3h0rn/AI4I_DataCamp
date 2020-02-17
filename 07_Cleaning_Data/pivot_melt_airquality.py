import pandas as pd
import os
os.chdir('/Users/apple/Documents/Online_Courses/DataCamp_Excercise/07_Cleaning_Data/')
airquality = pd.read_csv('airquality.csv')

# Print the head of airquality
print("original",airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(frame=airquality, id_vars=[
                          'Month', 'Day'], var_name='measurement', value_name='reading')

# Print the head of airquality_melt
print("melt", airquality_melt.head())


# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(
    index=['Month', 'Day'], columns='measurement', values='reading')

# Print the head of airquality_pivot
print("pivot", airquality_pivot.head())

# Print the index of airquality_pivot
print("pivotindex", airquality_pivot.index)
print("original index", airquality.index)

# Reset the index of airquality_pivot: airquality_pivot_reset
airquality_pivot_reset = airquality_pivot.reset_index()

# Print the new index of airquality_pivot_reset
print("reset index", airquality_pivot_reset.index)

# Print the head of airquality_pivot_reset
print("reset",airquality_pivot_reset.head())
