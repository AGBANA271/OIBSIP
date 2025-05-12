# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 16:48:30 2025

@author: USER
"""
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#load dataset
ifood_data = pd.read_csv('ifood_df.csv')

#Data Exploration and Cleaning
print(ifood_data.info())
print(ifood_data.describe())
print(ifood_data.isnull().sum())

#Check for any infinite value
#print(np.isinf(x).sum())

#Descriptive Statistics
#Total Spend
ifood_data['TotalSpend'] = ifood_data[['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)

#Drop Customers who never purchased
ifood_data = ifood_data[(ifood_data['NumWebPurchases'] > 0) &
                        (ifood_data['NumCatalogPurchases'] > 0) &
                        (ifood_data['NumStorePurchases'] > 0)]
#Average purchase value

# Now safe to create new columns
ifood_data = ifood_data.copy()
ifood_data['PurchaseFrequency'] = ifood_data[['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']].sum(axis=1)

ifood_data['AveragePurchaseValue'] =ifood_data['TotalSpend']/ifood_data['PurchaseFrequency']

#List of key metrics
key_metrics = ['TotalSpend','AveragePurchaseValue','PurchaseFrequency','Income','Recency']

#Create a figure wuth specified size
plt.Figure(figsize=(25,15))

#Loop through each metric and create a boxplot
flierprops = {"Markerfacecolor":"red", "markeredgeclor":"red"}
for i, metric in enumerate(key_metrics,1):
    plt.subplot(2,3,i)
    sn.boxplot(data=ifood_data, y=metric, color='skyblue')
    plt.title(f'Boxplot of {metric}', fontsize = 8)
    plt.suptitle('Boxplot of key_metric')
    plt.tight_layout(pad=3)    
plt.show()

#Distribution plots using Histogram
for i, metric in enumerate(key_metrics,1):
    plt.subplot(2,3,i)
    sn.histplot(data=ifood_data, x=metric, kde=True, color='salmon')
    plt.title(f'Distribution of {metric}', fontsize = 8)
    plt.tight_layout(pad=3)
plt.show()

#Functions to detect outliers based on IQR
def detect_outliers_iqr(ifood_data,column):
    Q1 = ifood_data[column].quantile(0.25)
    Q3 = ifood_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ifood_data[(ifood_data[column]<lower_bound) | (ifood_data[column]>upper_bound)]
    return outliers

#Detect outliers for each metrics
outliers_summary = {}
for metric in key_metrics:
    outliers_summary[metric] = detect_outliers_iqr(ifood_data, metric)
    
#Print number of outliers for each metric
for metric, outlier_ifood_data in outliers_summary.items():
    print(f'{metric}:{outlier_ifood_data.shape[0]} outliers detected')
          
#Top 5 Spenders
outliers_summary['TotalSpend'].sort_values(by = 'TotalSpend', ascending = False).head()

#Customer Segmentation (Clustering)
#Feature selection for Segmentation
features = ['Income', 'Recency', 'TotalSpend', 'PurchaseFrequency', 'AveragePurchaseValue']
#Define 'X"
x=ifood_data[features]

#Feature Scaling
Scaler = StandardScaler()
x_scaled = Scaler.fit_transform(x)

#Find number of clusters(Elbow method)
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, marker = 'o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()

#Apply kMeans Clustering
optimal_clusters = 4
kmeans = KMeans(n_clusters = optimal_clusters, random_state = 42)
ifood_data['cluster'] = kmeans.fit_predict(x_scaled)

#Visualize clusters with PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x_scaled)
ifood_data['PCA1'] = principal_components[:,0]
ifood_data['PCA2'] = principal_components[:,1]
plt.figure(figsize = (10,6))
sn.scatterplot(x='PCA1', y = 'PCA2', hue = 'cluster', data = ifood_data,palette = 'Set2')
plt.title('Customer Segment Visualization')
plt.show()

#Bar chart:Average metrics per cluster
#Select metrics to show
metrics = ['TotalSpend','AveragePurchaseValue','PurchaseFrequency','Income','Recency']

#Group by cluster
cluster_summary = ifood_data.groupby('cluster')[metric].mean().reset_index()

#Plot each metric
plt.figure(figsize = (20,12))
for i, metrics in enumerate(metric,1):
    plt.subplot (3,3,i)
    sn.barplot(x = 'cluster', y = metric, data=cluster_summary, palette = 'Set2')
    plt.title(f'Average{metric} by cluster')
    plt.tight_layout()
plt.show ()

#RInsights & Recommendation
"""
Based on Cluster
Cluster 0: High Income, High Spend, Frequent Purchases
Insights
- These are the premium loyal customers
-They buy often, spend a lot, and recently interacted
Recommendation:
    * Reward loyalty through VIP programs, exclusive discounts.
    * Upsell premium products or invite them to referral programs.
    * Maintain engagement with personalised marketing
    
Cluster 1: Medium Income, Medium Spend, Medium Recency
Insight
- Moderate in all metrics
Recommendation:
    * Use cross-sell campaigns or bundle deals.
    Educate them on premium products or loyalty programs

Cluster 2: Low Income, Low Spend, Infrequent and Less Recent
Insight
-These are at-risk or dormant customers
Recommendations:
    * Run re-engagement campaigns (e.g, "We miss you!") emails
    * Offer first purchase discounts to nudge action
    * Consider if low engagement is due to pricing- explore budget product lines

Cluster 3: High Income, Low Spend, Low Frequency
Insight
-These are opportunity customers: high potential but currently inactive
Recommendations:
    * Find out why they are not spending(survey or feedbackmloop).
    * Personalise offers based on demographics and showcase value
    * Launch a premium trial offer to engage.
    """











