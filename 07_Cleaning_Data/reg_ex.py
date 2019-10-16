import re

# Write the first pattern
# A telephone number of the format xxx-xxx-xxxx. You already did this in a previous exercise.
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)

# Write the second pattern
# A string of the format: A dollar sign, an arbitrary number of digits, a decimal point, 2 digits.
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
print(pattern2)

# Write the third pattern
# A capital letter, followed by an arbitrary number of alphanumeric characters.
pattern3 = bool(re.match(pattern='\[A-Z]\w*', string='Australia'))
print(pattern3)
