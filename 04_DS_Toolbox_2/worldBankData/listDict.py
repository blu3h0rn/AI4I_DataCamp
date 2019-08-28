
feature_names = ['CountryName', 'CountryCode',
                 'IndicatorName', 'IndicatorCode', 'Year', 'Value']
row_vals = ['Arab World', 'ARB',
            'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'SP.ADO.TFRT', '1960', '133.56090740552298']

#Create a zip object by calling zip() and passing to it feature_names and row_vals. 
# Assign the result to zipped_lists.

zipped_lists = zip(feature_names, row_vals)

#Create a dictionary from the zipped_lists zip object by calling dict() with zipped_lists. 
# Assign the resulting dictionary to rs_dict.
rs_dict = dict(zipped_lists)

# Print the dictionary
print(rs_dict)
