# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:17:52 2025

@author: USER
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

#Load data
TwitterData = pd.read_csv('Twitter_Data.csv')

#Drop rows
TwitterData = TwitterData.dropna(subset = ['clean_text','category'])
TwitterData['category'] = TwitterData['category'].astype(int)
print(TwitterData.columns.tolist())

#Sample to reduce processing time
TwitterData_sampled = TwitterData.sample(n=20000, random_state = 42)

#Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    TwitterData_sampled['clean_text'],
    TwitterData_sampled['category'], 
    test_size = 0.2, 
    random_state = 42                                                    
)

#Build and train model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words= 'english', max_features = 3000)),
    ('clf', MultinomialNB())
    ])
pipeline.fit(x_train, y_train)

#Predict and map sentiment
TwitterData_sampled['predicted_sentiment']=pipeline.predict(TwitterData_sampled['clean_text'])
sentiment_map = {-1: 'Negative', 0:'Neutral',  1:'Positive'}
TwitterData_sampled['sentiment_label'] = TwitterData_sampled['predicted_sentiment'].map(sentiment_map)

#Initialize Dash App
app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
app.title = "Twitter Sentment Dashboard"

#Predict Sentiment using your trained pipeline
TwitterData_sampled['predicted_sentiment']=pipeline.predict(TwitterData_sampled['clean_text'])

#Map predicted labels to text
sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
TwitterData_sampled['sentiment_label'] = TwitterData_sampled['predicted_sentiment'].map(sentiment_map)

#Pie chart figure
fig = px.pie(
    TwitterData_sampled,
    names = 'sentiment_label',
    title = 'Predicted Sentiment Distribution',
    color = 'sentiment_label',
    color_discrete_map= {'Positive': 'green', 'Neutral': 'grey', 'Negative': 'red'}
    )
#fig.show()

print( TwitterData_sampled.head())
print( TwitterData_sampled['sentiment_label'].unique())

#Layout
app.layout = dbc.Container([
    html.H2("Twitter Sentiment Analysis Dashboard", className="my-4"),
    dcc.Graph(figure=fig),
    dbc.Row([
        dbc.Col([
            html.Label("search tweets by keyword:"),
            dcc.Input(id='keyword-input', type='text', 
placeholder ='e.g., modi, election', debounce = True),
        ], width=6)
    ], className = "my-3"),
    html.Div(id='tweet-output')
], fluid=True)

#Callback to update tweet list based on Keyword
@app.callback(
    Output('tweet-output', 'children'),
    Input('keyword-input', 'value')
    )
def update_tweets(keyword):
    if keyword is None or keyword == '':
        return html.Div("Please enter a keyword")
    
    #filter tweets by keyword
    filtered = TwitterData_sampled[
        TwitterData_sampled['clean_text'].str.contains(keyword, case=False, na=False)
        ]
    if filtered.empty:
        return html.Div("No tweets found with that keyword", className="text-danger")
    else:
        samples = filtered[['clean_text', 'sentiment_label']].head(5).values.tolist()
        return html.Ul([
            html.Li([
                html.Strong(f"[{sentiment}]"),
                ":",
                text
                ]) for text, sentiment in samples
])
    
import webbrowser

if __name__=='__main__':
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=True)










