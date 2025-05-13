# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:28:48 2025

@author: USER
"""


import json 
import pandas as pd

#Load dataset
with open('CA_category_id.json') as f:
    CA_category = json.load(f)

#Extract video categories from json data
if 'items' in CA_category:
    video_categories = CA_category['items']

    #cleaned the data
    cleaned_CA_category = []   
    for category in video_categories:
        if 'id' in category and 'snippet' in category:
            if 'title' in category['snippet'] and 'assignable' in category['snippet']:
                cleaned_category = {
                    'id': category['id'],
                    'title': category['snippet']['title'],
                    'assignable': category['snippet']['assignable']
                }
                cleaned_CA_category.append(cleaned_category)
 
#Create DataFrame                   
    CA_category = pd.DataFrame(cleaned_CA_category)

#Get information about DataFrame
    
    CA_category.info()
    CA_category.describe()
    print(CA_category)
    print(CA_category.shape)
    print(CA_category.isnull().sum())
else:
    print("No 'items' key found in the Json data.")
              
#CA_category.info()
#CA_category.describe()
##print(CA_category.shape)
#print(CA_category.isnull().sum())
