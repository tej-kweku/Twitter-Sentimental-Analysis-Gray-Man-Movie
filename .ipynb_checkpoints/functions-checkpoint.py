import urllib.request
import urllib.parse
import json
from collections import Counter
import re
import string
import time
import os
import random
import csv # to read and write csv files
import glob # to retrieve files/pathnames matching a specified pattern. 
import sys
import datetime as dt

import tweepy
import streamlit as st
import pandas as pd
import numpy as np

from textblob import TextBlob

import nltk

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords # get stopwords from NLTK library
from nltk.tokenize import word_tokenize # to create word tokens
from nltk.stem.porter import * # (I played around with Stemmer and decided to use Lemmatizer instead)
from nltk.stem import WordNetLemmatizer # to reduce words to orginal form
from nltk import pos_tag # For Parts of Speech tagging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import words # Get all words in english language

# from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO

from sklearn.feature_extraction.text import CountVectorizer

from scheduler import Scheduler
import scheduler.trigger as trigger


tweets_file_name = "tweets_grayman.csv"
current_file_name = "tweets_grayman.csv"
latest_file_name = "tweets_grayman_latest.csv"
miner_log_file_name = "miner_log.txt"
analysis_log_file_name = "./analysis_log.txt"
base_path = "."

here_URL = "https://geocode.search.hereapi.com/v1/geocode"  # Deevloper Here API link
here_api_key = 'dZutQ0xR3yZaTGC4Ws5BHUKpzYoaSx6awHEW0aOAMAY'  # Acquire api key from developer.here.com

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


# The list below are common words which will not be relevant in our analysis.
common_words = ['netflix']
alphabets = list(string.ascii_lowercase)
stop_words = stop_words + alphabets + common_words + grayman_characters
#word_list = words.words()  # all words in English language
# emojis = list(UNICODE_EMOJI.keys())


os.chdir(base_path)


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
            location = tweet.user.location # retrieve user location
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

    
# get the location data
def get_coordinates(location):
    PARAMS = {'apikey': here_api_key, 'q': location} # required parameters
    r = requests.get(url=here_URL, params=PARAMS)  # pass in required parameters
    # get raw json file. I did  this because when I combined this step with my "getLocation" function, 
    # it gave me error for countries with no country_code or country_name. Hence, I needed to use try - except block
    data = r.json() # Raw json file 
    return data


# a function to extract required coordinates information to the tweets_df dataframe
def get_location(location):
    for data in location:
        if len(location['items']) > 0:
            try:   
                country_code = location['items'][0]['address']['countryCode']
                country_name = location['items'][0]['address']['countryName']
            except KeyError:
                country_code = float('Nan')
                country_name = float('Nan')
        else: 
            country_code = float('Nan') 
            country_name = float('Nan')
        result = (country_code, country_name)
    return result


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
            # hashtags.append(i)
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
    # filtered_words = [w for w in filtered_words if w not in emojis]
    
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
    # print(sentiments_df)
    
    return {
        "general": general_df,
        "sentiments": sentiments_df
    }
    
   
# @st.cache(allow_output_mutation=True)
def preprocess_tweets():
    # print("inner: ", updated_ts)
    try:
        cols = ["tweet_id", "created_at", "text", "location", "retweet", "favorite"]
        tweets_df = pd.read_csv(current_file_name, header=None, index_col=None, names=cols)

    #     tweets_df['location_data'] = tweets_df['location'].apply(get_coordinates)
    #     tweets_df['country_name_code'] = tweets_df['location_data'].apply(getLocation) #apply getLocation function

    #     # Extraction of Country names to different columns
    #     tweets_df[['country_code','country_name']] = pd.DataFrame(tweets_df['country_name_code'].tolist(),index=tweets_df.index)
    #     # Drop unnecessary columns
    #     tweets_df.drop(['location','location_data','country_name_code'], axis = 1, inplace = True)

        # Rename columns
        # tweets_df.columns = ['tweet_id','time_created','tweet','retweet_count','favorite_count','country_code','country_name']
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

        # tweets_df.to_csv("processed_tweets_grayman.csv", encoding="utf-8", index=False)
    except FileNotFoundError as err:
        tweets_df = pd.DataFrame()

    return tweets_df


# def load_tweets():
#     updated_ts = time.ctime(os.path.getmtime(current_file_name))
#     st.write(updated_ts)
    
#     return preprocess_tweets()

