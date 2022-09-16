import tweepy
import pandas as pd
import csv 
import re 
import glob 
import os
import time
import sys
import datetime as dt
import string

from textblob import TextBlob
import demoji

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

print("MINER: ", os.getcwd().upper())


consumer_key = "gEro0S6tWFM0H1ujdND2lLemF"
consumer_secret = "Ij2eXRiA6lzzSaHnb6EAcVJvKs3jXd2Tj1WUhVmvxP3KGasd5G"
access_token = "1536381857033601025-PaZbQo2eRMwgWn1fyyyvxHLopX0nrT"
access_token_secret = "EFsBgaJlZ1cQqzpXnSyqrFuxeyjeApMbA7ds4KfYy8K4x"

current_file_name = "tweets_grayman.csv"
latest_file_name = "tweets_grayman_latest.csv"

miner_log_file_name = "miner_log.txt"

# relevant data for NLTK 
grayman_characters = ["six", "lloyd", "hansen", "dani", "miranda", "fitzroy", "suzanne", "brewer", "avik", "san", "margaret", "cahill", "carmichael", "laszlo", "sosa", "claire", "father", "dulin", "perini", "markham", "dining", "car", "buyer", "young", "dawson", "officer", "zelezny"]
stop_words = list(stopwords.words('english'))

# The list below are common words which will not be relevant in our analysis.
common_words = ['netflix']
alphabets = list(string.ascii_lowercase)
stop_words = stop_words + alphabets + common_words + grayman_characters

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret) 
api = tweepy.API(auth,wait_on_rate_limit=True) 


def extract_hashtags(tweet):
    if pd.isna(tweet):
        return ""
    else:
        tweet = str(tweet).lower()  #has to be in place
        tweet = re.findall(r'\#\w+',tweet) # Remove hastags with REGEX
        return " ".join(tweet)


# Function to extract movie Characters from each Tweet
def extract_movie_characters(tweet):
    if pd.isna(tweet):
        return ""
    else:
        tweet = tweet.lower() # Reduce tweet to lower case
        tweet_tokens = word_tokenize(tweet) # split each word in the tweet for parsing
        movie_characters = [char for char in tweet_tokens if char in grayman_characters] # extract movie characters
        return " ".join(movie_characters).strip()



def refine_tweet_text(tweet):
    if pd.isna(tweet):
        return ""
    else:
        tweet = tweet.lower()
        # remove emojis 
        tweet = demoji.replace(tweet, "")
        # Remove urls
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags = re.MULTILINE)
        # Remove user @ references and '#' from tweet
        tweet = re.sub(r'\@\w+|\#\w+|\d+', '', tweet)
        tweet_tokens = word_tokenize(tweet)  # convert string to tokens
        # Remove stopwords
        filtered_words = [w for w in tweet_tokens if w not in stop_words]

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


def log(log_text, where="miner"):
    if where == "miner":
        log_file_name = miner_log_file_name
    elif where == "analysis":
        log_file_name = analysis_log_file_name
        
    print(log_text)
    with open(log_file_name,'a', newline='', encoding='utf-8') as log_file:
        log_file.write(log_text)


def get_tweets(search_query, num_tweets, period="current", since_id_num=None):
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
                since_id=since_id_num,
                tweet_mode='extended'
            ).items(num_tweets)
        ]
        file_name = latest_file_name
        
    file_name = file_name    
    tweets = []
    for tweet in tweet_list:
        tweet_id = tweet.id
        created_at = tweet.created_at
        text = tweet.full_text
        location = tweet.user.location 
        retweet = tweet.retweet_count
        favorite = tweet.favorite_count 

        tweets.append([tweet_id, created_at, text, location, retweet, favorite])
        
    cols = ["tweet_id", "created_at", "tweet", "location", "retweet", "favorite"]
    tweets_df = pd.DataFrame(tweets, columns=cols)
    
    # extract hashtags
    tweets_df['hashtags'] = tweets_df['tweet'].apply(extract_hashtags)
    # extract movie characters
    tweets_df['movie_characters'] = tweets_df['tweet'].apply(extract_movie_characters)
    # refine tweet text
    tweets_df['tweet_refined'] = tweets_df['tweet'].apply(refine_tweet_text)
    # get sentiments
    tweets_df['polarity']=tweets_df['tweet_refined'].apply(get_polarity)
    tweets_df['sentiment']=tweets_df['polarity'].apply(get_sentiment_textblob)
    
    tweets_df.to_csv(file_name, header=True, index=False)
        
    
    log_text = "{}: {}\t {}\t{} tweet(s)\n".format(time.asctime(), period, file_name, len(tweet_list))
    log(log_text, "miner")
        

def mine():
    if (len(sys.argv) > 1) and (sys.argv[1] == "restart"):
        if (os.path.exists(miner_log_file_name) and os.path.getsize(miner_log_file_name) > 0):
            os.remove(miner_log_file_name)
        for file_name in glob.glob("*.csv"):
            os.remove(file_name)
        print("Restarting")
        log_text = "{}: {}".format(time.asctime(), "restarting\n")
        log(log_text, "miner")

    search_words = "thegrayman OR grayman OR thegreyman OR greyman OR ryangosling OR chrisevans OR sierra6 OR #thegrayman OR #grayman OR #thegreyman OR #greyman OR #ryangosling OR #chrisevans OR #sierra6"
    search_query = search_words + " -filter:retweets AND -filter:replies"

    # get the current tweets
    if not os.path.exists("tweets_grayman.csv"):
        print("No tweets mined previously. Fetching...") 
        get_tweets(search_query,15000)
    else:
        print("Found previously mined tweets")

    with open('tweets_grayman.csv', encoding='utf-8') as data_file:
        next(csv.reader(data_file))  # skip the header line
        latest_tweet = next(csv.reader(data_file))
        latest_tweet_id = int(latest_tweet[0].strip())

    print("Fetching latest tweets")
    get_tweets(search_query, 5000, "latest", latest_tweet_id)


    tweets = []
    for file_name in [ current_file_name, latest_file_name ]:
        df = pd.read_csv(file_name, index_col=None, header="infer")
        print(df.shape)
        tweets.append(df)

    tweets_df = pd.concat(tweets)
    tweets_df = tweets_df.sort_values(by=["tweet_id"], ascending=False).drop_duplicates()
    
    tweets_df.to_csv("tweets_grayman.csv", header=True, index=False)
    os.remove(latest_file_name)

    
if __name__ == "__main__":
    mine()
    
    schedule = Scheduler()
    schedule.cyclic(dt.timedelta(minutes=180), mine) 
        
    while True:
        schedule.exec_jobs()
        time.sleep(1)