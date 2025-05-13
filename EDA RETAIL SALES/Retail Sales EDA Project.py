# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:56:58 2025

@author: USER
"""

# retail Sales EDA Project

# Import Libraries 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

#Loading Retail Sales Data

RetailSales = pd.read_csv('retail_sales_dataset.csv')
MenuData = pd.read_csv('menu.csv')

# Data cleaning
#display first few rows

display(RetailSales.head())
display(MenuData.head())

# check for missing values
RetailSales.isnull
MenuData.isnull

#Descriptive Statistics
def basic_stats(RetailSales, column):
    print(f"\n Basic Statistics for '{column}':")
    print(f"mean:{RetailSales[column].mean()}")
    print(f"median:{RetailSales[column].median()}")
    print(f"mode:{RetailSales[column].mode().values}")
    print(f"standard deviation:{RetailSales[column].std():.2f}")
    
basic_stats(RetailSales, 'Quantity')
basic_stats(RetailSales, 'Price per Unit')
basic_stats(RetailSales, 'Total Amount')

#print("\n Descriptive Stats(Dataset 2):")
#print(MenuData.describe())

# Time series analysis
# Convert to datetime
RetailSales['Date'] = pd.to_datetime(RetailSales['Date'])

#Sales trends over time

Time_Series = RetailSales.groupby(RetailSales['Date'].dt.to_period('M'))['Total Amount'].sum()

#chart for monthly sales trend

plt.figure(figsize = (10,5))
Time_Series.plot(kind ='line', marker = 'o', color = 'teal')
plt.title('Monthly Sales Trend', fontsize = 14)
plt.xlabel('Month')
plt.ylabel ('Total Sales')
plt.show()

#Customer and Product Analysis
#Top 10 customers
top_customers = RetailSales.groupby('Customer ID')['Total Amount'].sum().sort_values(ascending = False).head(10)
top_customers.plot(kind='bar', title = 'Top 10 Customers by Total Amount')
plt.show()

#Product Analysis
top_products = RetailSales.groupby('Product Category')['Total Amount'].sum().sort_values(ascending = False)
top_products.plot(kind='bar', title = 'Top Products by Total Amount')
plt.show()

#Pivot and heatmap
#Extract Month from Date column and add it as a new column

RetailSales['Date'] = pd.to_datetime(RetailSales['Date'])
RetailSales['Month'] = RetailSales['Date'].dt.month_name()

# Create a Pivot Table
pivot_RetailSales = pd.pivot_table(RetailSales,
values = 'Total Amount', index = 'Product Category', 
columns = 'Month', aggfunc='sum')

#Plot the heatmap
plt.figure(figsize=(10,8))
sn.heatmap(pivot_RetailSales, annot=True, cmap='Blues',fmt='d')
plt.title('RetailSales Heatmap')
plt.xlabel('Month')
plt.ylabel('Product Category')
plt.show()

#Insights & Recommendation:
"""
1. Sales Performance over time
Insight
-The monthly trend line shows how total sales trends over time
Actionable Recommendations:
    * Identify high performing months and investigate what marketing promotions
    or external factors contributed to those peaks-replicate or refinforce these
    strategies.
    * Address low-performing months with targeed discounts,marketing campaigns, 
    or new product launches to boost engagement.
    * Comsider seasonal demand planning to optimize inventory and staffing
2. Top Customers
Insight
- A small set of customers contributes significantly to revenue
Actionable Recommendations:
    * Reward top customers through loyalty programs, early access to new 
    products or personalised offers to retain them
    * Analyse the characteristics of top spenders(location, product preference,
    frequency) and find similar customer profiles to target through ads.
    * Consider setting up customers tiers to encourage more spending
3. Product Category Analysis
Insight
- Some product categories dominate in revenue
Actionable Recommendations:
    * Prioritise top-performing categories in marketing and stock replenishment.
    *Discontinue or review low-performing categories-rebrand, bundle or phase them.
4. Monthly Product Sales Heatmap 
Insight
- Certain product categories peakin specific months
Actionable Recommendations:
    * use the heatmap to identify peak sales  periods for specific products and
    djust inventory and marketing strategies accordingly
    *Analyse the heatmap to determine which prduct categories are performing well
    during specific months and adjust marketing efforts to capitalise on these
    trends.
"""



