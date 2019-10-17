#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:52:13 2019

@author: apple
"""

#------------------------------
# Loading and exploring a JSON
#------------------------------

# Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ',json_data[k])

#---------------
# API requests
#---------------
    
# Import requests package
import requests

# Assign URL to variable: url
url = "http://www.omdbapi.com/?apikey=72bc447a&t=the+social+network"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Print the text of the response
print(r.text)

#------------------------------
# JSON–from the web to Python
#------------------------------

# Import package
import requests

# Assign URL to variable: url
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=social+network'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Decode the JSON data into a dictionary: json_data
json_data = r.json()

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])


#--------------------------------
# Checking out the Wikipedia API
#--------------------------------


# Import package
import requests

# Assign URL to variable: url
url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza"

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Decode the JSON data into a dictionary: json_data
json_data = r.json()


# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)
