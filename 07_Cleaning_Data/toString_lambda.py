import pandas as pd
import numpy as np

tips = pd.read_csv('tips.csv')

# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: x.re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())