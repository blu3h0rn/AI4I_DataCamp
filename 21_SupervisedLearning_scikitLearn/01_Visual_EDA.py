import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

file= "/Users/apple/Documents/Online_Courses/DataCamp_Excercise/21_SupervisedLearning_scikitLearn/house-votes-84.csv"
df = pd.read_csv(file)

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()