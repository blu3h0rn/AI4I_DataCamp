import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house-votes-84.csv")

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()