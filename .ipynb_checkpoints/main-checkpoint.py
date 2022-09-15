import time
import os
import sys
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import warnings

from streamlit_autorefresh import st_autorefresh

from functions import *

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')


# base_path = "."

# os.chdir(base_path)

print("MAIN: ", os.getcwd().upper())

st.experimental_singleton.clear()
st.experimental_memo.clear()
        
st.set_page_config(
    layout="wide", 
#     page_icon="", 
    page_title="Netflix-Twitter Sentimental Analysis"
)


# global variable to store OMDB movie data
omdb_data = None
# global variable to store Twitter meta data
twitter_meta_data = None
# global variable to store Twitter data
twitter_data = None


def reload_data():
    twitter_data = preprocess_tweets()
        
# fetch the movie data
with st.spinner("Loading"):
    # raw_data = fetch_movie_data_from_internet()
    raw_data = movie_data
    omdb_data = json.loads(raw_data)

    twitter_data = preprocess_tweets()        
    twitter_meta_data = dict()    

    with st.sidebar:
        st.subheader("Filters")

        today = datetime.date.today()
        prev_day = datetime.date(2022, 8, 1)
        start_date = st.date_input('Start date', prev_day)
        end_date = st.date_input('End date', today)

        sel_hashtags = [ x[1:] for x in get_hashtags(twitter_data)["hashtags"].values ]
        selected_hashtags = st.multiselect(
            "Hashtags",
            sel_hashtags
        )

        selected_sentiment = st.multiselect(
            "Sentiment",
            ("Negative", "Neutral", "Positive")
        )

        sel_movie_characters = [ x for x in get_movie_characters(twitter_data)["movie_characters"].values ]
        selected_movie_character = st.multiselect(
            "Movie Characters",
            sel_movie_characters
        )

        if start_date:
            twitter_data = twitter_data.loc[
                (twitter_data["time_created"].dt.date >= start_date)
            ]
        if end_date:
            twitter_data = twitter_data.loc[
                (twitter_data["time_created"].dt.date <= end_date)
            ]
        if len(selected_hashtags) > 0:
            twitter_data = twitter_data.loc[
                (twitter_data["hashtags"].str.contains("|".join(selected_hashtags)))
            ]
        if len(selected_sentiment) > 0:
            twitter_data = twitter_data.loc[
                (twitter_data["sentiment"].str.contains("|".join(selected_sentiment)))
            ]
        if len(selected_movie_character) > 0:
            twitter_data = twitter_data.loc[
                (twitter_data["movie_characters"].str.contains("|".join(selected_movie_character))) 
            ]

twitter_meta_data["tweets"] = str(twitter_data.shape[0])
twitter_meta_data["retweets"] = str(sum([ int(x["retweet_count"]) for idx, x in twitter_data.iterrows() ]))
twitter_meta_data["likes"] = str(sum([ int(x["favorite_count"]) for idx, x in twitter_data.iterrows() ]))
twitter_meta_data["hashtags"] = get_hashtags(twitter_data)
twitter_meta_data["movie_characters"] = get_movie_characters(twitter_data)
twitter_meta_data["daily_report"] = get_daily_report(twitter_data)


netflix_logo_link = "https://www.edigitalagency.com.au/wp-content/uploads/Netflix-logo-red-black-png-1200x681.png"
font_awesome_link = """<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/all.min.css">"""
st.write(font_awesome_link, unsafe_allow_html=True)

left_col, right_col = st.columns([1, 6])
with left_col:
    st.image(netflix_logo_link)
with right_col:
    st.header("Twitter Sentimental Analysis")
    st.header("Trending Movies")

space(2)


c1 = st.container()    
c1_1_left_col, c1_1_middle_col, c1_1_right_col = c1.columns([2, 2, 6])

c1_1_left_col.markdown("<p style='text-align: center; vertical-align: center; margin-bottom: 0px'>Tweets </p>", unsafe_allow_html=True)
c1_1_left_col.markdown("<h1 style='text-align: center; vertical-align: center; margin:0px 0px 50px 0px'; padding-top:0px>" + twitter_meta_data["tweets"] + " <i class='fa-solid fa-xs fa-comment-alt'></i></h1>", unsafe_allow_html=True)
space(2)

c1_1_left_col.markdown("<p style='text-align: center; vertical-align: center;; margin-bottom: 0px'>Retweets</p>", unsafe_allow_html=True)
c1_1_left_col.markdown("<h1 style='text-align: center; vertical-align: center; margin:0px 0px 50px 0px'; padding-top:0px>" + twitter_meta_data["retweets"] + " <i class='fa-solid fa-xs fa-retweet'></i></h1>", unsafe_allow_html=True)
space(2)

c1_1_left_col.markdown("<p style='text-align: center; vertical-align: center;; margin-bottom: 0px'>Likes</p>", unsafe_allow_html=True)
c1_1_left_col.markdown("<h1 style='text-align: center; vertical-align: center; margin:0px 0px 50px 0px'; padding-top:0px>" + twitter_meta_data["likes"] + "<i class='fa-solid fa-xs fa-heart'></i></h1>", unsafe_allow_html=True)

c1_1_middle_col.image(omdb_data["Poster"])

c1_1_right_col.subheader(omdb_data["Title"])
c1_1_right_col.write(omdb_data["Plot"])
c1_1_right_col.write("Director:\t" + omdb_data["Director"])
c1_1_right_col.write("Genre:\t" + omdb_data["Genre"])
c1_1_right_col.write("Actors:\t" + omdb_data["Actors"])
c1_1_right_col.write("Year:\t" + omdb_data["Year"])
c1_1_right_col.write("Runtime:\t" + omdb_data["Runtime"])
c1_1_right_col.write("Country:\t" + omdb_data["Country"])

c2 = st.container()
c2_left_col, c2_middle_col, c2_right_col = c2.columns([3, 3, 3])
with c2_left_col:
    c2_left_col.subheader("Top 10 Hashtags")
    fig = px.bar(
        twitter_meta_data["hashtags"][:10], 
        x="count", 
        y="hashtags", 
        labels={"hashtags": "Hashtag", "count": "Mentions"}
    )
    c2_left_col.plotly_chart(fig, use_container_width=True)

with c2_middle_col:
    c2_middle_col.subheader("Sentiments")
    sentiments_df = get_sentiments(twitter_data)
    fig = px.pie(
        sentiments_df, 
        values='Value', 
        names='Sentiments',
        hole=0.3,
        color_discrete_map={'Negative':'red','Positive':'green'}
    )
    c2_middle_col.plotly_chart(fig, use_container_width=True)

with c2_right_col:
    c2_right_col.subheader("Most Mentioned Characters")
    fig = px.bar(
        twitter_meta_data["movie_characters"][:10], 
        x="count", 
        y="movie_characters", 
        labels={"movie_characters": "Movie Character", "count": "Mentions"}
    )
    c2_right_col.plotly_chart(fig, use_container_width=True)

space(2)

c3 = st.container()    
c3_left_col, c3_right_col = c3.columns([6, 3])
with c3_left_col:
    c3_left_col.subheader("Trend")  
    c3_left_col_tab_1, c3_left_col_tab_2 = c3_left_col.tabs(["General", "Sentiment"])
    with c3_left_col_tab_1:
        fig = px.line(
            twitter_meta_data["daily_report"]["general"], 
            x="day", 
            y="Count", 
            color="Type", 
            markers=True,
            labels={"day": "Date"},
            height=500
        )
        c3_left_col_tab_1.plotly_chart(fig, use_container_width=True)

    with c3_left_col_tab_2:
        fig = px.line(
            twitter_meta_data["daily_report"]["sentiments"], 
            x="day", 
            y="Count", 
            color="sentiment", 
            markers=True,
            labels={"day": "Date"},
            height=500
        )
        c3_left_col_tab_2.plotly_chart(fig, use_container_width=True)


with c3_right_col:
    c3_right_col.subheader("Tweets")
    c3_right_col.dataframe(twitter_data.sample(100)["tweet"])


log_text = "{}: Reload Completed {} tweet(s)\n".format(time.asctime(), twitter_data.shape[0])
print(log_text)
log(log_text, "analysis")

reload_interval = 1800000  # every 30 minuntes
count = st_autorefresh(interval=reload_interval, key="tsagrayman")
