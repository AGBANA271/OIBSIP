# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:21:06 2025

@author: USER
"""

#Import library
import pandas as pd

#Load dataset
AB_NYC = pd.read_csv('AB_NYC_2019.csv')

#overview
print("Initial shape:", AB_NYC.shape)
AB_NYC.info()
AB_NYC.describe(include = 'all')

#missing values
print("missing values before cleaning:\n", AB_NYC.isnull().sum())

#check for duplicates
#duplicates = AB_NYC.duplicated().sum()
#print(f"Number of duplicate rows:{duplicates}")

#drop duplicates
AB_NYC = AB_NYC.drop_duplicates()

#fill missing host_name with "Unknown'
AB_NYC['host_name'] = AB_NYC['host_name'].fillna('Unknown')

#to view the host_name column
#print(AB_NYC['host_name'].value_counts())

#check for 'Unknown' values
#unknown_hosts = AB_NYC[AB_NYC['host_name'] == 'Unknown']
#print(unknown_hosts)

#fill missing reviews_per_month with 0
AB_NYC['reviews_per_month'] = AB_NYC['reviews_per_month'].fillna(0)

#drop rows only where 'name' is missing and 'last_review')
AB_NYC = AB_NYC.dropna(subset = ['name', 'last_review'])

#convert 'last_review' to datetime 
AB_NYC['last_review'] = pd.to_datetime(AB_NYC['last_review'])

#remove rows with price < 0
AB_NYC = AB_NYC[AB_NYC['price']>0]

#cap outliers in 'minimum_night' using IQR
Q1 = AB_NYC['minimum_nights'].quantile(0.25)
Q3 = AB_NYC['minimum_nights'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR
AB_NYC = AB_NYC[AB_NYC['minimum_nights'] <= upper_limit]

#final check
print(AB_NYC.shape)
print(AB_NYC.isnull().sum())


