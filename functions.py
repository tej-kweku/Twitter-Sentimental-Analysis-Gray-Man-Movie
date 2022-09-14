import urllib.request
import urllib.parse
import json
from collections import Counter
import re
import string
import time
import os
import random
import csv 
import glob 
import sys
import datetime as dt

import tweepy
import streamlit as st
import pandas as pd
import numpy as np

from textblob import TextBlob
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO

import nltk

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import * 
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import words 

from sklearn.feature_extraction.text import CountVectorizer

from scheduler import Scheduler
import scheduler.trigger as trigger


movie_data = """{"Title":"The Gray Man","Year":"2022","Rated":"PG-13","Released":"22 Jul 2022","Runtime":"122 min","Genre":"Action, Thriller","Director":"Anthony Russo, Joe Russo","Writer":"Joe Russo, Christopher Markus, Stephen McFeely","Actors":"Ryan Gosling, Chris Evans, Ana de Armas","Plot":"When the CIA's most skilled operative-whose true identity is known to none-accidentally uncovers dark agency secrets, a psychopathic former colleague puts a bounty on his head, setting off a global manhunt by international assassins.","Language":"English","Country":"United States, Czech Republic","Awards":"N/A","Poster":"https://m.media-amazon.com/images/M/MV5BOWY4MmFiY2QtMzE1YS00NTg1LWIwOTQtYTI4ZGUzNWIxNTVmXkEyXkFqcGdeQXVyODk4OTc3MTY@._V1_SX300.jpg","Ratings":[{"Source":"Internet Movie Database","Value":"6.5/10"},{"Source":"Rotten Tomatoes","Value":"46%"},{"Source":"Metacritic","Value":"49/100"}],"Metascore":"49","imdbRating":"6.5","imdbVotes":"134,431","imdbID":"tt1649418","Type":"movie","DVD":"22 Jul 2022","BoxOffice":"N/A","Production":"N/A","Website":"N/A","Response":"True"}"""


tweets_file_name = "tweets_grayman.csv"
current_file_name = "tweets_grayman.csv"
latest_file_name = "tweets_grayman_latest.csv"
miner_log_file_name = "miner_log.txt"
analysis_log_file_name = "analysis_log.txt"
base_path = "."

consumer_key = "gEro0S6tWFM0H1ujdND2lLemF"
consumer_secret = "Ij2eXRiA6lzzSaHnb6EAcVJvKs3jXd2Tj1WUhVmvxP3KGasd5G"
access_token = "1536381857033601025-PaZbQo2eRMwgWn1fyyyvxHLopX0nrT"
access_token_secret = "EFsBgaJlZ1cQqzpXnSyqrFuxeyjeApMbA7ds4KfYy8K4x"
bearer_token = "AAAAAAAAAAAAAAAAAAAAAFCLgAEAAAAAFbl7j0Y3EYiuXAevDNsxXeIjTS8%3DszUHqfnB503qY6VvJS9tYkkmYAxbeyH7WHp3sWUe5tgXraH8mb"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

tweepy_client = tweepy.Client(bearer_token)

tweets_df = None
tweets_long_string = None

# relevant data for NLTK 

grayman_characters = ["six", "lloyd", "hansen", "dani", "miranda", "fitzroy", "suzanne", "brewer", "avik", "san", "margaret", "cahill", "carmichael", "laszlo", "sosa", "claire", "father", "dulin", "perini", "markham", "dining", "car", "buyer", "young", "dawson", "officer", "zelezny"]
stop_words = list(stopwords.words('english'))
emojis = list(UNICODE_EMOJI.keys()) 


# The list below are common words which will not be relevant in our analysis.
common_words = ['netflix']
alphabets = list(string.ascii_lowercase)
stop_words = stop_words + alphabets + common_words + grayman_characters

# os.chdir(base_path)


def log(log_text, where="miner"):
    if where == "miner":
        log_file_name = miner_log_file_name
    elif where == "analysis":
        log_file_name = analysis_log_file_name
        
    with open(log_file_name,'a', newline='', encoding='utf-8') as log_file:
        log_file.write(log_text)
                

def get_tweets(search_query, num_tweets, period="current", max_id_num=None, since_id_num=None):
    file_name = None
    if period == "current":
        tweet_list = [
            tweets for tweets in tweepy.Cursor(
                api.search_tweets,
                q=search_query,
                lang="en", 
                tweet_mode='extended'
            ).items(num_tweets)
        ]
        file_name = current_file_name
    elif period == "latest":
        tweet_list = [
            tweets for tweets in tweepy.Cursor(
                api.search_tweets,
                q=search_query,
                lang="en",
                since_id=since_id_num, # since_id is the most recent tweet id you have
                tweet_mode='extended'
            ).items(num_tweets)
        ]
        file_name = latest_file_name
    file_name = file_name
    with open(file_name,'a', newline='', encoding='utf-8') as csvFile:
        csv_writer = csv.writer(csvFile, delimiter=',') # create an instance of csv object
        # Begin scraping the tweets individually:
        for tweet in tweet_list:
            tweet_id = tweet.id # get Tweet ID result
            created_at = tweet.created_at # get time tweet was created
            text = tweet.full_text # retrieve full tweet text
            retweet = tweet.retweet_count # retrieve number of retweets
            favorite = tweet.favorite_count # retrieve number of likes
            
            csv_writer.writerow([tweet_id, created_at, text, location, retweet, favorite])
        log_text = "{}: {}\t {}\t{} tweet(s)\n".format(time.asctime(), period, file_name, len(tweet_list))
        log(log_text, "miner")
        print("Mining completed")
             

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")
        
        
def fetch_movie_data_from_internet():
    params = urllib.parse.urlencode({"t": "gray man", "apikey": "3870044d"})
    url = "http://www.omdbapi.com/?%s" % params
    with urllib.request.urlopen(url) as f:
        data = f.read().decode('utf-8')
        
        return data
        

def percentage(part,whole):
    return 100 * float(part)/float(whole)


def extract_hashtags(tweet):
    tweet = tweet.lower()  #has to be in place
    tweet = re.findall(r'\#\w+',tweet) # Remove hastags with REGEX
    return " ".join(tweet)


def get_hashtags(tweets_df):
    search_words = "thegrayman OR grayman OR thegreyman OR greyman OR ryangosling OR chrisevans OR sierra6 OR #thegrayman OR #grayman OR #thegreyman OR #greyman OR #ryangosling OR #chrisevans OR #sierra6"
    hashtags_list = tweets_df['hashtags'].tolist()
    hashtags = []

    for item in hashtags_list:
        item = item.split()
        for i in item:
            if i in search_words:
                hashtags.append(i)

    counts = Counter(hashtags)
    hashtags_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    hashtags_df.columns = ['hashtags', 'count']
    hashtags_df.sort_values(by='count', ascending=False, inplace=True)
    hashtags_df["percentage"] = 100*(hashtags_df["count"]/hashtags_df['count'].sum())

    return hashtags_df


# Function to extract movie Characters from each Tweet
def extract_movie_characters(tweet):
    tweet = tweet.lower() # Reduce tweet to lower case
    tweet_tokens = word_tokenize(tweet) # split each word in the tweet for parsing
    movie_characters = [char for char in tweet_tokens if char in grayman_characters] # extract movie characters
    return " ".join(movie_characters).strip()


def get_movie_characters(tweets_df):
    characters_list = tweets_df['movie_characters'].tolist()

    characters = []
    for item in characters_list:
        item = item.split()
        for i in item:
            characters.append(i)

    counts = Counter(characters)
    characters_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    characters_df.columns = ['movie_characters', 'count']
    characters_df.sort_values(by='count', ascending=False, inplace=True)
    characters_df['percentage'] = 100*(characters_df['count'] / characters_df['count'].sum())
        
    return characters_df


def refine_tweet_text(tweet):
    tweet = tweet.lower()  # changes all words to lower case
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags = re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#\w+|\d+', '', tweet)
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)  # convert string to tokens
    filtered_words = [w for w in tweet_tokens if w not in stop_words]
    filtered_words = [w for w in filtered_words if w not in emojis]
    
    # Remove punctuations
    unpunctuated_words = [w for w in filtered_words if w not in string.punctuation]
    lemmatizer = WordNetLemmatizer() # instatiate an object WordNetLemmatizer Class
    lemma_words = [lemmatizer.lemmatize(w) for w in unpunctuated_words]
    return " ".join(lemma_words)


# Create function to obtain Polarity Score
def get_polarity(tweet):
    return TextBlob(tweet).sentiment.polarity

# Create function to obtain Sentiment category
def get_sentiment_textblob(polarity):
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"


def get_sentiments(tweets_df):
    sentiments_df = tweets_df['sentiment'].value_counts().rename_axis('Sentiments').to_frame('Value').reset_index()
    
    return sentiments_df


def get_daily_report(tweet_df):
    tweet_df["day"] = tweet_df["time_created"].dt.date
    general_df = tweet_df.groupby(["day"]).agg(
        {"tweet": "count", "retweet_count": "sum", "favorite_count": "sum"}
    ).reset_index().rename(columns={"index": "day"})
    
    general_df = pd.melt(general_df, id_vars=["day"], value_vars=["tweet", "retweet_count", "favorite_count"], var_name="Type", value_name="Count")
    general_df["Type"] = general_df["Type"].replace({
        "tweet": "Tweet",
        "retweet_count": "Retweet",
        "favorite_count": "Like"
    })
    
    sentiments_df = tweet_df.groupby(by=["day", "sentiment"]).count()["polarity"]
    sentiments_df = sentiments_df.reset_index().rename(columns={"polarity": "Count"})
    
    return {
        "general": general_df,
        "sentiments": sentiments_df
    }
    
   
def preprocess_tweets():
    try:
        cols = ["tweet_id", "created_at", "text", "location", "retweet", "favorite"]
        tweets_df = pd.read_csv(current_file_name, header=None, index_col=None, names=cols)

        # Rename columns
        tweets_df.columns = ['tweet_id','time_created','tweet', 'location', 'retweet_count','favorite_count']

        tweets_df['time_created'] = pd.to_datetime(tweets_df['time_created'])

        # extract hashtags
        tweets_df['hashtags'] = tweets_df['tweet'].apply(extract_hashtags)

        # extract movie characters
        tweets_df['movie_characters'] = tweets_df['tweet'].apply(extract_movie_characters)

        # refine tweet text
        tweets_df['tweet_refined'] = tweets_df['tweet'].apply(refine_tweet_text)
        tweets_long_string = tweets_df['tweet_refined'].tolist()
        tweets_long_string = " ".join(tweets_long_string)

        tweets_df['polarity']=tweets_df['tweet_refined'].apply(get_polarity)
        tweets_df['sentiment']=tweets_df['polarity'].apply(get_sentiment_textblob)
    except FileNotFoundError as err:
        tweets_df = pd.DataFrame()

    return tweets_df