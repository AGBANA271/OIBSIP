# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:43:44 2025

@author: USER
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sn

#Load data
HousePrices = pd.read_csv('Housing.csv')

#info about the dataset
print("Dataset Info:")
print(HousePrices.info())

#Display first few rows
print("\n Sample Rows:")
print(HousePrices.head())

#Check for missing values
print("\n Missing Values:")
print(HousePrices.isnull().sum())

#Summary statistics for numerical features
print("\n Summary Statistics(Numerical):")
print(HousePrices.describe())

#Count unique values for categorical feaures
categorical_cols = HousePrices.select_dtypes(include='object').columns
print("\n Categorical Feature Summary:")
for col in categorical_cols:
    print(f"{col} Value Counts:")
    display(HousePrices[col].value_counts())
    ##print(HousePrices[col].value_counts())
    #print()
    
#Feature Selection
#Convert categorical variables to numerical using one-hot encoding
HousePrices_encoded = pd.get_dummies(HousePrices, drop_first=True)

#Separate the target variable (price) and features
x = HousePrices_encoded.drop("price", axis=1)
y = HousePrices_encoded["price"]

#Display the new feature set columns
print("Selected Features:")
print(x.columns.tolist())

#Split into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

#Initialize and train the model
model = LinearRegression()
model.fit(x_train, y_train)

#Print model coefficients and intercept
coefficients = pd.Series(model.coef_, index=x.columns)
intercepts = model.intercept_

print(" Model Coefficients:")
print(coefficients)
print("\n Model Intercepts:")
print(intercepts)

#Predict on the test set
y_pred = model.predict(x_test)

#Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Print results
print("Model Evaluation Metrics:")
print(f"Mean Squared Error(MSE): {mse:,.2f}")
print(f"R-squared (R²): {r2: .4f}" )

#Visualization of actuals vs predicted plot
plt.figure(figsize=(6,6))
sn.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label="Perfect Prediction")
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Residual plot
residuals = y_test - y_pred

#Create residual plot
plt.figure(figsize=(6,4))
sn.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Predicted Price')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True)
plt.tight_layout()
plt.show()











    
     
    
    
    
    
    
    
    
    