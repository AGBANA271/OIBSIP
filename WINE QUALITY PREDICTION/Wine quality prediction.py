# -*- coding: utf-8 -*-
"""
Created on Sat May  3 19:01:49 2025

@author: USER
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np



#Load dataset

wine_quality = pd.read_csv('WineQT.csv')

#Display first few rows
print(wine_quality.head())

#Dataset structure and data types
print(wine_quality.info())

#Basic statistics
print(wine_quality.describe())

#Checking for missing values and duplicates

#Check for Null values
print("\nMissing Values:\n", wine_quality.isnull().sum())

#Check for duplicates
duplicate_count = wine_quality.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_count}")

#Visualize Distributions and Relationships
#Histogram of all numeric features
wine_quality.hist(bins=15, figsize=(15,10))
plt.suptitle("Features Distributions", fontsize=16)
plt.tight_layout()
plt.show()

#Correlation  heatmap
plt.figure(figsize=(10,8))
sn.heatmap(wine_quality.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#Explore Quality Distribution
#Count plot for wine quality
sn.countplot(x="quality", data=wine_quality)
plt.title("Wine Quality Distribution")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.show()

#Feature Correlation with Quality
#Correlation with target variable 'quality'
correlation_with_quality = wine_quality.corr()['quality'].sort_values(ascending=False)
print("\nCorrelation with wine quality:\n", correlation_with_quality)

#Train Classifiers Model
#Preprocessing: Features and Target
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Separate features and target
x = wine_quality.drop('quality', axis=1)
y = wine_quality["quality"]

#Train-test split(80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Scale features for models like SVC and SGD
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Model Training and Evaluation  Function
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)

print("Random Forest Performance:")
evaluate_model(rf_model, x_test, y_test)

#Support Vector Classifier(SVC)
from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(x_train_scaled, y_train)
print("SVC Performance:")
evaluate_model(svc_model, x_test_scaled, y_test)

#Stochastic Gradient Descent(SGD)

from sklearn.linear_model import SGDClassifier

sgd_model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
sgd_model.fit(x_train_scaled, y_train)

print("SGD Classification Performance")
evaluate_model(sgd_model, x_test_scaled, y_test)

#Scatter plot: Density Vs Quality
sn.scatterplot(x='density', y='quality', data = wine_quality)
plt.title("Density vs Wine Quality")
plt.show()

#Scatter plot: Volatle Acidity vs Quality
sn.scatterplot(x='volatile acidity', y='quality', data = wine_quality)
plt.title("Volatile Acidity vs Wine Quality")
plt.show()

#Scatter plot: Citric Acid vs Quality
sn.scatterplot(x='citric acid', y='quality', data=wine_quality)
plt.title("Citric Acid vs Wine Quality")
plt.show()

#Boxplot: Density by Quality
sn.boxplot(x='quality', y='density', data=wine_quality)
plt.title("Density Across Quality Levels")
plt.show()

#Boxplot: Fixed Acidity by Quality
sn.boxplot(x='quality', y='fixed acidity', data=wine_quality)
plt.title("Fixed Acidity Across Quality Levels")
plt.show()

#Create a new column: acidity_ratio = fixed_acidity / volatile_acidity
wine_quality['acidity_ratio'] = wine_quality['fixed acidity'] / wine_quality['volatile acidity']

#Normalize 'alcohol' using Numpy (min-max scaling)
alcohol = wine_quality['alcohol'].values
alcohol_norm = (alcohol - np.min(alcohol)) / (np.max(alcohol) - np.min(alcohol))
wine_quality['alcohol_normalized'] = alcohol_norm

#Categorize wine quality: low (3-4), medium (5-6), high(7-8)
wine_quality['quality_category'] = pd.cut(wine_quality['quality'],
                                          bins=[2,4,6,8],
                                          labels=['low', 'medium', 'high'])

#Preview the changes
print(wine_quality[['fixed acidity', 'volatile acidity', 'acidity_ratio', 
                    'alcohol', 'alcohol_normalized', 'quality_category']].head())































