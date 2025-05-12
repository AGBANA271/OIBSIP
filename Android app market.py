# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:41:08 2025

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

android_app = pd.read_csv('Apps.csv')

android_app.info()

print(android_app.isnull().sum())

#Drop the irrelevant index column
android_app.drop(columns=["Unnamed: 0"], inplace=True)

#clean 'Installs: remove commas and plus signs, then convert to integer
android_app['Installs'] = (
    android_app['Installs']
    .astype(str)
    .str.replace('[+,]', '', regex=True) #remove commas and plus sign
    .str.strip()   
)

#Convert to integers, coercing errors to NaN if needed
android_app['Installs'] = pd.to_numeric(android_app['Installs'], errors='coerce')

#Drop rows where Installs couldn't be parsed (Optional):
android_app =  android_app.dropna(subset=['Installs'])
android_app['Installs'] = android_app['Installs'].astype(int)    
    
#Clean 'Price': remove dollar signs and convert to float
android_app['Price'] = (
    android_app['Price']
    .astype(str) #ensure all values are strings
    .str.replace('$',  '', regex=True)
    .str.strip() #Remove any leading/trailing whitespace
)

#Convert to float, set errors='coerce to handle non-numeric values
android_app['Price'] = pd.to_numeric(android_app['Price'], errors='coerce')

#Convert 'Size' to MB: handle 'M', 'R', and 'Varies with device'
def convert_size(size_str):
    if isinstance(size_str, str):
        size_str = size_str.strip()
        if size_str.endswith('M'):
            return float(size_str.replace('M', ''))
        elif size_str.endswith('k'):
            return float(size_str.replace('k', ''))/1024
        return np.nan
    android_app['Size_MB'] = android_app['Size'].apply(convert_size)

#Drop rows where critical values are missing(Rating, Type, Size)
android_app_clean = android_app.dropna(subset=['Rating', 'Type', 'Size'])

#Reset index after cleaning
android_app_clean.reset_index(drop=True, inplace=True)

#Visualize Average Rating per Category
#Calculate average rating per category
avg_rating_per_category = android_app.groupby('Category')['Rating'].mean().sort_values(ascending=False)

#Plot
plt.figure(figsize=(14,8))
sn.barplot(x=avg_rating_per_category.values, y=avg_rating_per_category.index, palette="coolwarm")

plt.title('Average App Rating per Category', fontsize=16)
plt.xlabel('Average Rating')
plt.ylabel('Category')
plt.xlim(3.5, 5.0)
plt.tight_layout()
plt.show()

#Metrics Analysis Code(Rating vs Size, Price, Installs)
#Rating vs App Size
plt.figure(figsize=(12,6))
sn.scatterplot(data=android_app_clean, x='Size', y='Rating', alpha=0.6)
plt.title('Rating vs. App Size (MB)')
plt.xlabel('App Size (MB)')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()

#Ratings vs. Price(All apps )
plt.figure(figsize=(12,6))
sn.scatterplot(data=android_app_clean, x='Price', y='Rating', color='tomato', alpha=0.6)
plt.title('Rating vs. Price (All Apps')
plt.xlabel('Price ($)')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()

print("Number of paid apps:",
      android_app_clean[android_app_clean['Price'] > 0].shape[0])

#Rating vs, Installs (Log scale)
plt.figure(figsize=(12,6))
sn.scatterplot(
    data=android_app_clean,
    x=np.log(android_app_clean['Installs']),
    y='Rating',
    color='seagreen',
    alpha=0.6
 )
plt.title('Rating vs. Installs')
plt.xlabel('Log Installs')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()

#How to check the minimum value
print(android_app_clean['Installs'].min())

#Sentiment Analysis
import nltk
nltk.download('punkt')

import pandas as pd
from textblob import TextBlob
import seaborn as sn
from wordcloud import WordCloud
import matplotlib.pyplot as plt


#Load the review data
UserReviews = pd.read_csv('user_reviews.csv')

#Drop rows with missing sentiment or reviews
UserReviews_clean= UserReviews.dropna(subset=["Sentiment", "Translated_Review"])

#sentiment distribution
sentiment_counts = UserReviews_clean["Sentiment"].value_counts()

#Plot sentiment distribution
plt.figure(figsize=(8,5))
sn.barplot(x=sentiment_counts.index, y=sentiment_counts, palette="Set2")
plt.title("Sentiment Distribution")
plt.ylabel("Number of Reviews")
plt.xlabel("Sentiment")
plt.show()

#Generate word clouds for each sentiment
for sentiment in ["Positive", "Negative", "Neutral"]:
    text = " ".join(UserReviews_clean[UserReviews_clean["Sentiment"]==sentiment]["Translated_Review"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {sentiment} Reviews")
    plt.show()


def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

#Apply sentiment analysis
UserReviews['Sentiment_Predicted'] = UserReviews['Translated_Review'].apply(get_sentiment)

#Visualize sentiment distribution
sn.countplot(data=UserReviews, x='Sentiment_Predicted', palette='pastel')
plt.title('User Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()

print(UserReviews_clean.columns)

#Top 10 most reviewed apps
review_counts = UserReviews_clean['App'].value_counts().head(10)

#Plot Top 10 most reviewed apps
plt.figure(figsize=(10,6))
sn.barplot(x=review_counts.values, y=review_counts.index, palette='coolwarm')
plt.title('Top 10 Most Reviewed Apps')
plt.xlabel('Number of Reviews')
plt.ylabel('Apps')
plt.show()

#Top 5 most reviewed apps for sentiment breakdown
top_apps = review_counts.index[:5]
top_apps_UserReviews = UserReviews_clean[UserReviews_clean['App'].isin(top_apps)]

#Sentiment breakdown
plt.figure(figsize=(12,6))
sn.countplot(data=top_apps_UserReviews, y='App', hue='Sentiment', order=top_apps, palette='Set2')
plt.title('Sentiment Breakdown for Top 5 Most Reviewed Apps')
plt.xlabel('Number of Reviews')
plt.ylabel('Apps')
plt.legend(title='Sentiment')
plt.show()


















