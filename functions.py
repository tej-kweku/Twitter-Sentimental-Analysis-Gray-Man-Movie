from collections import Counter

import streamlit as st
import pandas as pd

st.experimental_singleton.clear()
st.experimental_memo.clear()


movie_data = {"Title":"The Gray Man","Year":"2022","Rated":"PG-13","Released":"22 Jul 2022","Runtime":"122 min","Genre":"Action, Thriller","Director":"Anthony Russo, Joe Russo","Writer":"Joe Russo, Christopher Markus, Stephen McFeely","Actors":"Ryan Gosling, Chris Evans, Ana de Armas","Plot":"When the CIA's most skilled operative-whose true identity is known to none-accidentally uncovers dark agency secrets, a psychopathic former colleague puts a bounty on his head, setting off a global manhunt by international assassins.","Language":"English","Country":"United States, Czech Republic","Awards":"N/A","Poster":"https://m.media-amazon.com/images/M/MV5BOWY4MmFiY2QtMzE1YS00NTg1LWIwOTQtYTI4ZGUzNWIxNTVmXkEyXkFqcGdeQXVyODk4OTc3MTY@._V1_SX300.jpg","Ratings":[{"Source":"Internet Movie Database","Value":"6.5/10"},{"Source":"Rotten Tomatoes","Value":"46%"},{"Source":"Metacritic","Value":"49/100"}],"Metascore":"49","imdbRating":"6.5","imdbVotes":"134,431","imdbID":"tt1649418","Type":"movie","DVD":"22 Jul 2022","BoxOffice":"N/A","Production":"N/A","Website":"N/A","Response":"True"}


current_file_name = "tweets_grayman.csv"
miner_log_file_name = "miner_log.txt"
analysis_log_file_name = "analysis_log.txt"
base_path = "."


tweets_df = None
tweets_long_string = None


def log(log_text, where="miner"):
    if where == "miner":
        log_file_name = miner_log_file_name
    elif where == "analysis":
        log_file_name = analysis_log_file_name
        
    with open(log_file_name,'a', newline='', encoding='utf-8') as log_file:
        log_file.write(log_text)
                             

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")
        
        

def percentage(part,whole):
    return 100 * float(part)/float(whole)



def get_hashtags(tweets_df):
    search_words = "thegrayman OR grayman OR thegreyman OR greyman OR ryangosling OR chrisevans OR sierra6 OR #thegrayman OR #grayman OR #thegreyman OR #greyman OR #ryangosling OR #chrisevans OR #sierra6"
    hashtags_list = tweets_df['hashtags'].tolist()
    hashtags = []

    for item in hashtags_list:
        if type(item) == str:
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


def get_movie_characters(tweets_df):
    characters_list = tweets_df['movie_characters'].tolist()

    characters = []
    for item in characters_list:
        if type(item) == str:
            item = item.split()
            for i in item:
                characters.append(i)

    counts = Counter(characters)
    del counts["NA"]
    characters_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    characters_df.columns = ['movie_characters', 'count']
    characters_df.sort_values(by='count', ascending=False, inplace=True)
    characters_df['percentage'] = 100*(characters_df['count'] / characters_df['count'].sum())
        
    return characters_df


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
    
   
def read_dataset():
    try:
        tweets_df = pd.read_csv(current_file_name, header="infer", index_col=None)

        # Rename columns
        tweets_df.columns = ['tweet_id','time_created','tweet', 'location', 
            'retweet_count', 'favorite_count', 'hashtags', 'movie_characters', 
            'tweet_refined', 'polarity', 'sentiment']

        tweets_df['time_created'] = pd.to_datetime(tweets_df['time_created'], errors="coerce")
        
        tweets_long_string = tweets_df['tweet_refined'].tolist()
        tweets_df["hashtags"] = tweets_df["hashtags"].fillna("NA")
        tweets_df["movie_characters"] = tweets_df["movie_characters"].fillna("NA")
    except FileNotFoundError as err:
        tweets_df = pd.DataFrame()

    return tweets_df