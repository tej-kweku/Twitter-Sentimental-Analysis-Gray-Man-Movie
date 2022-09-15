import tweepy
import pandas as pd
import csv 
import re 
import glob 
import os
import time
import sys
import datetime as dt

from scheduler import Scheduler
import scheduler.trigger as trigger

print("MINER: ", os.getcwd().upper())


consumer_key = "gEro0S6tWFM0H1ujdND2lLemF"
consumer_secret = "Ij2eXRiA6lzzSaHnb6EAcVJvKs3jXd2Tj1WUhVmvxP3KGasd5G"
access_token = "1536381857033601025-PaZbQo2eRMwgWn1fyyyvxHLopX0nrT"
access_token_secret = "EFsBgaJlZ1cQqzpXnSyqrFuxeyjeApMbA7ds4KfYy8K4x"

current_file_name = "tweets_grayman.csv"
latest_file_name = "tweets_grayman_latest.csv"

# base_path = "."
miner_log_file_name = "miner_log.txt"

# os.chdir(base_path)

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret) 
api = tweepy.API(auth,wait_on_rate_limit=True) 

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
    with open(file_name,'a', newline='', encoding='utf-8') as csvFile:
        csv_writer = csv.writer(csvFile, delimiter=',') 
        for tweet in tweet_list:
            tweet_id = tweet.id
            created_at = tweet.created_at
            text = tweet.full_text
            location = tweet.user.location 
            retweet = tweet.retweet_count
            favorite = tweet.favorite_count 
            
            csv_writer.writerow([tweet_id, created_at, text, location, retweet, favorite])
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
        first_tweet = next(csv.reader(data_file))
        latest_tweet_id = int(first_tweet[0])

    print("Fetching latest tweets")
    get_tweets(search_query, 5000, "latest", latest_tweet_id)


    tweets = []
    cols = ["tweet_id", "created_at", "text", "location", "retweet", "favorite"]
    for file_name in [ current_file_name, latest_file_name ]:
        df = pd.read_csv(file_name, index_col=None, header=None, names=cols) 
        print(df.shape)
        tweets.append(df)

    tweets_df = pd.concat(tweets)
    tweets_df = tweets_df.sort_values(by=["tweet_id"], ascending=False).drop_duplicates()

    tweets_df.to_csv("tweets_grayman.csv", header=False, index=False)

    
if __name__ == "__main__":
    mine()
    
    schedule = Scheduler()
    schedule.cyclic(dt.timedelta(minutes=360), mine) 
        
    while True:
        schedule.exec_jobs()
        time.sleep(1)
        
        
