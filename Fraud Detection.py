# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:20:04 2025

@author: USER
"""

#import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV 

#Load the dataset
credit_card = pd.read_csv('Creditcard.csv')

print(credit_card.info())

#Preprocess the data
x = credit_card.drop(['Class'], axis=1) 
y = credit_card['Class']

#Scale the data using StandardScaler
scaler =  StandardScaler()
x_scaled = scaler.fit_transform(x)

#Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

#Handle imbalance classes using SMOTE oversampling
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

#samle down while testing
x_sample = x_train_resampled[:10000]
y_sample = y_train_resampled[:10000]
 
#Define base model
model = RandomForestClassifier(random_state = 42)

#Define hyperparameter tuning space for RandomForestClassifier 
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
    }

#Perform hyperparameter using RandomizedSearchCV with verbosity
random_search = RandomizedSearchCV(
    estimator = model, 
    param_distributions = param_grid, 
    n_iter=5, 
    cv=3, 
    n_jobs=-1,
    verbose=2,
    random_state=42
 )
random_search.fit(x_train_resampled, y_train_resampled)

#Get the best-performing model and its parameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_

#Evaluate the best model on the test set
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification_report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Use the best model to detect fraud in new transactions
def detect_fraud(transaction):
    feature_columns = credit_card.drop(['Class'], axis=1).columns.tolist()
    transaction_credit_card = pd.DataFrame([transaction], columns=feature_columns)
    transaction_scaled = scaler.transform(transaction_credit_card)
    prediction = best_model.predict(transaction_scaled)
 
    return "Fraud detected"  if prediction[0] == 1 else "Legitimate transaction"
    

transaction = credit_card.drop(['Class'], axis=1).iloc[0].tolist()
print(detect_fraud(transaction))

#View all fraud transactions
fraud_transactions = credit_card[credit_card['Class'] == 1]
print(fraud_transactions)

#View only the first few (top 5)
print(fraud_transactions.head())















