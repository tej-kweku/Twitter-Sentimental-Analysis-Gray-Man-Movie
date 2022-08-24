import tweepy # for tweet mining
import pandas as pd
import csv # to read and write csv files
import re # In-built regular expressions library
import glob # to retrieve files/pathnames matching a specified pattern. 
import os
import time
import sys


consumer_key = "gEro0S6tWFM0H1ujdND2lLemF"
consumer_secret = "Ij2eXRiA6lzzSaHnb6EAcVJvKs3jXd2Tj1WUhVmvxP3KGasd5G"
access_token = "1536381857033601025-PaZbQo2eRMwgWn1fyyyvxHLopX0nrT"
access_token_secret = "EFsBgaJlZ1cQqzpXnSyqrFuxeyjeApMbA7ds4KfYy8K4x"

current_file_name = "tweets_grayman.csv"
older_file_name = "tweets_grayman_older.csv"
latest_file_name = "tweets_grayman_latest.csv"

base_path = "/home/tejkweku/Personal Studies/Udacity/ALx-T Data Science/Career Support/Project/v3/"
miner_log_file_name = "miner_log.txt"

os.chdir(base_path)

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) # Pass in Consumer key and secret for authentication by API
auth.set_access_token(access_token, access_token_secret) # Pass in Access key and secret for authentication by API
api = tweepy.API(auth,wait_on_rate_limit=True) # Sleeps when API limit is reached


def log(log_text):
    with open(miner_log_file_name,'a', newline='', encoding='utf-8') as log_file:
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
    elif period == "older":
        tweet_list = [
            tweets for tweets in tweepy.Cursor(
                api.search_tweets,
                q=search_query,
                lang="en",
                max_id=max_id_num, # max_id is the oldest tweet id you have
                tweet_mode='extended'
            ).items(num_tweets)
        ]
        file_name = older_file_name
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
    with open("tmp.txt", "w", newline="\n", encoding="utf-8") as tmpFile:
        tmpFile.write(str(tweet_list[:5]))
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
        log(log_text)
        

if (len(sys.argv) > 1) and (sys.argv[1] == "restart"):
    os.remove(miner_log_file_name)
    for file_name in glob.glob("*.csv"):
        os.remove(file_name)
    print("Restarting")
    log_text = "{}: {}".format(time.asctime(), "restarting\n")
    log(log_text)
            
search_words = "thegrayman OR grayman OR thegreyman OR greyman OR #thegrayman OR #grayman OR #thegreyman OR #greyman"
# Exclude Links, retweets, replies
search_query = search_words + " -filter:retweets AND -filter:replies"

# get the current tweets
if not os.path.exists("tweets_grayman.csv"):
    print("No tweets mined previously. Fetching...") 
    get_tweets(search_query,100)
else:
    print("Found previously mined tweets")

with open('tweets_grayman.csv', encoding='utf-8') as data:
    tweets = list(csv.reader(data))
    # oldest_tweet_id = int(tweets[-1][0]) # Return the oldest tweet ID
    latest_tweet_id = int(tweets[0][0]) # Return the latest tweet ID
    

# if not os.path.exists("tweets_grayman_older.csv"):
#     print("Fetching older tweets")
#     get_tweets(search_query, 10, "older", None, oldest_tweet_id)
    
print("Fetching latest tweets")
get_tweets(search_query, 100, "latest", None, latest_tweet_id)


tweets = []
cols = ["tweet_id", "created_at", "text", "location", "retweet", "favorite"]
for file_name in glob.glob("*.csv"):
    if (os.path.exists(file_name) and os.path.getsize(file_name) > 0):
        # Convert each csv to a dataframe
        print(file_name, os.path.getsize(file_name))
        df = pd.read_csv(file_name, index_col=None, header=None, names=cols) 
        tweets.append(df)
        
tweets_df = pd.concat(tweets)#, axis=0, ignore_index=True) # Merge all dataframes
tweets_df = tweets_df.sort_values(by=["tweet_id"], ascending=False).drop_duplicates()

# for file_name in glob.glob("*.csv"):
#     os.remove(file_name)

# os.remove(*latest_file_name)

tweets_df.to_csv("tweets_grayman.csv", header=False, index=False)
