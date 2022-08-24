import tweepy # for tweet mining
import pandas as pd
import numpy as np
import csv # to read and write csv files
import re # In-built regular expressions library
import string # Inbuilt sting library
import glob # to retrieve files/pathnames matching a specified pattern. 
import random # generating random numbers
import requests # to send HTTP requests
import os
from PIL import Image # for opening, manipulating, and saving many different image file f
import matplotlib.pyplot as plt # for plotting


#Text Sentiments
#TextBlob - Python library for processing textual data
import textblob
from textblob import TextBlob 

# #Visualization tools
# import plotly.express as px # To make express plots in Plotly
# import chart_studio.tools as cst # For exporting to Chart studio
# import chart_studio.plotly as py # for exporting Plotly visualizations to Chart Studio
# import plotly.offline as pyo # Set notebook mode to work in offline
# pyo.init_notebook_mode()
# import plotly.io as pio # Plotly renderer
# import plotly.graph_objects as go # For plotting plotly graph objects
# from plotly.subplots import make_subplots

# WordCloud - Python linrary for creating image wordclouds
from wordcloud import WordCloud
# from emot.emo_unicode import UNICODE_EMO, EMOTICONS # For emojis

consumer_key = "gEro0S6tWFM0H1ujdND2lLemF"
consumer_secret = "Ij2eXRiA6lzzSaHnb6EAcVJvKs3jXd2Tj1WUhVmvxP3KGasd5G"
access_token = "1536381857033601025-PaZbQo2eRMwgWn1fyyyvxHLopX0nrT"
access_token_secret = "EFsBgaJlZ1cQqzpXnSyqrFuxeyjeApMbA7ds4KfYy8K4x"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) # Pass in Consumer key and secret for authentication by API
auth.set_access_token(access_token, access_token_secret) # Pass in Access key and secret for authentication by API
api = tweepy.API(auth,wait_on_rate_limit=True) # Sleeps when API limit is reached


def get_tweets(search_query, num_tweets):
    tweet_list = [
        tweets for tweets in tweepy.Cursor(
            api.search_tweets,
            q=search_query,
            lang="en", 
            tweet_mode='extended'
        ).items(num_tweets)
    ]
    with open('tweets_grayman.csv','a', newline='', encoding='utf-8') as csvFile:
        csv_writer = csv.writer(csvFile, delimiter=',') # create an instance of csv object
        # Begin scraping the tweets individually:
        for tweet in tweet_list:
            tweet_id = tweet.id # get Tweet ID result
            created_at = tweet.created_at # get time tweet was created
            text = tweet.full_text # retrieve full tweet text
            location = tweet.user.location # retrieve user location
            retweet = tweet.retweet_count # retrieve number of retweets
            favorite = tweet.favorite_count # retrieve number of likes
            
            csv_writer.writerow([tweet_id, created_at, text, location, retweet, favorite])
            
search_words = "thegrayman OR grayman OR thegreyman OR greyman OR #thegrayman OR #grayman OR #thegreyman OR #greyman"
# Exclude Links, retweets, replies
search_query = search_words + " -filter:retweets AND -filter:replies"
get_tweets(search_query,500)

# THIS FUNCTION IS USED TO MINE TWEETS THAT ARE OLDER THAN THE TWEETS YOU HAVE.
def get_tweets_older(search_query, num_tweets, max_id_num):
    tweet_list = [
        tweets for tweets in tweepy.Cursor(
            api.search_tweets,
            q=search_query,
            lang="en",
            max_id=max_id_num, # max_id is the oldest tweet id you have
            tweet_mode='extended'
        ).items(num_tweets)
    ]
    with open('tweets_grayman_older.csv', 'a', newline='', encoding='utf-8') as csvFile:
        csv_writer = csv.writer(csvFile, delimiter=',')  # create an instance of csv object
        # Begin scraping the tweets individually:
        for tweet in tweet_list:
            tweet_id = tweet.id  # get Tweet ID result
            created_at = tweet.created_at  # get time tweet was created
            text = tweet.full_text  # retrieve full tweet text
            location = tweet.user.location  # retrieve user location
            retweet = tweet.retweet_count  # retrieve number of retweets
            favorite = tweet.favorite_count  # retrieve number of likes

            csv_writer.writerow([tweet_id, created_at, text, location, retweet, favorite])

            
# THIS FUNCTION IS USED TO MINE TWEETS THAT ARE NEWER THAN THE TWEETS YOU HAVE.
def get_tweets_latest(search_query, num_tweets, since_id_num):
    tweet_list = [
        tweets for tweets in tweepy.Cursor(
            api.search_tweets,
            q=search_query,
            lang="en",
            since_id=since_id_num, # since_id is the most recent tweet id you have
            tweet_mode='extended'
        ).items(num_tweets)
    ]
    with open('tweets_grayman_latest.csv','a', newline='', encoding='utf-8') as csvFile:
        csv_writer = csv.writer(csvFile, delimiter=',') # create an instance of csv object
        # Begin scraping the tweets individually:
        for tweet in tweet_list[::-1]:
            tweet_id = tweet.id # get Tweet ID result
            created_at = tweet.created_at # get time tweet was created
            text = tweet.full_text # retrieve full tweet text
            location = tweet.user.location # retrieve user location
            retweet = tweet.retweet_count # retrieve number of retweets
            favorite = tweet.favorite_count # retrieve number of likes
            
            csv_writer.writerow([tweet_id, created_at, text, location, retweet, favorite])

# get the current tweets
if not os.path.exists("tweets_grayman.csv"):
    print("No tweets mined previously. Fetching...") 
    get_tweets(search_query,500)

with open('tweets_grayman.csv', encoding='utf-8') as data:
    tweets = list(csv.reader(data))
    oldest_tweet_id = int(tweets[-1][0]) # Return the oldest tweet ID
    latest_tweet_id = int(tweets[0][0]) # Return the latest tweet ID
    

if not os.path.exists("tweets_grayman_older.csv"):
    print("Fetching older tweets"))
    get_tweets_older(search_query, 500, oldest_tweet_id)
print("Fetching latest tweets")
get_tweets_latest(search_query, 500, latest_tweet_id)


tweets = []
base_path = "."
cols = ["tweet_id", "created_at", "text", "location", "retweet", "favorite"]
for filename in glob.glob(base_path + "/*.csv"):
    if (os.path.exists(filename) and os.path.getsize(filename) > 0):
        # Convert each csv to a dataframe
        print(filename, os.path.getsize(filename))
        df = pd.read_csv(filename, index_col=None, header=None, names=cols) 
        tweets.append(df)

tweets_df = pd.concat(tweets)#, axis=0, ignore_index=True) # Merge all dataframes
tweets_df = tweets_df.sort_values(by=["tweet_id"]).drop_duplicates()

for filename in glob.glob(base_path + "/*.csv"):
    os.remove(filename)

tweets_df.to_csv("tweets_grayman.csv", header=False, index=False)
