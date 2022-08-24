import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import warnings
import time
import os
import sys

from functions import *

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')


base_path = "/home/tejkweku/Personal Studies/Udacity/ALx-T Data Science/Career Support/Project/v3/"


os.chdir(base_path)

        
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

try:

    # fetch the movie data
    with st.spinner("Loading"):
        # omdb_data = {'Title': 'The Gray Man', 'Year': '2022', 'Rated': 'PG-13', 'Released': '22 Jul 2022', 'Runtime': '122 min', 'Genre': 'Action, Thriller', 'Director': 'Anthony Russo, Joe Russo', 'Writer': 'Joe Russo, Christopher Markus, Stephen McFeely', 'Actors': 'Ryan Gosling, Chris Evans, Ana de Armas', 'Plot': "When the CIA's most skilled operative-whose true identity is known to none-accidentally uncovers dark agency secrets, a psychopathic former colleague puts a bounty on his head, setting off a global manhunt by international assassins.", 'Language': 'English', 'Country': 'United States, Czech Republic', 'Awards': 'N/A', 'Poster': 'https://m.media-amazon.com/images/M/MV5BOWY4MmFiY2QtMzE1YS00NTg1LWIwOTQtYTI4ZGUzNWIxNTVmXkEyXkFqcGdeQXVyODk4OTc3MTY@._V1_SX300.jpg', 'Ratings': [{'Source': 'Internet Movie Database', 'Value': '6.5/10'}, {'Source': 'Rotten Tomatoes', 'Value': '46%'}, {'Source': 'Metacritic', 'Value': '49/100'}], 'Metascore': '49', 'imdbRating': '6.5', 'imdbVotes': '134,431', 'imdbID': 'tt1649418', 'Type': 'movie', 'DVD': '22 Jul 2022', 'BoxOffice': 'N/A', 'Production': 'N/A', 'Website': 'N/A', 'Response': 'True'}

        raw_data = fetch_movie_data_from_internet()
        omdb_data = json.loads(raw_data)
        # # print(json.loads(raw_data).keys()
        # print(json.loads(raw_data))
        # twitter_meta_data = movie_twitter_metadata("gray man")

        twitter_data = preprocess_tweets()
        # st.dataframe(twitter_data)
        # print(twitter_data.info())

        twitter_meta_data = dict()    
        twitter_meta_data["tweets"] = str(twitter_data.shape[0])
        twitter_meta_data["retweets"] = str(np.sum([ int(x["retweet_count"]) for idx, x in twitter_data.iterrows() ]))
        twitter_meta_data["likes"] = str(np.sum([ int(x["favorite_count"]) for idx, x in twitter_data.iterrows() ]))
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
    # space(2)

    c1_1_middle_col.image(omdb_data["Poster"])

    c1_1_right_col.subheader(omdb_data["Title"])
    c1_1_right_col.write(omdb_data["Plot"])
    c1_1_right_col.write("Director:\t" + omdb_data["Director"])
    c1_1_right_col.write("Genre:\t" + omdb_data["Genre"])
    c1_1_right_col.write("Actors:\t" + omdb_data["Actors"])
    # c1_1_right_col.write("Ratings:\t" + str(omdb_data["Ratings"]))
    c1_1_right_col.write("Year:\t" + omdb_data["Year"])
    c1_1_right_col.write("Runtime:\t" + omdb_data["Runtime"])
    c1_1_right_col.write("Country:\t" + omdb_data["Country"])


    # space(2)

    c2 = st.container()
    c2_left_col, c2_middle_col, c2_right_col = c2.columns([3, 3, 3])
    with c2_left_col:
        # c2_left_col.markdown("##### Top 10 #Hashtags")
        c2_left_col.subheader("Top 10 Hashtags")
        # st.markdown("""---""")
        fig = px.bar(
            twitter_meta_data["hashtags"][:10], 
            x="count", 
            y="hashtags", 
            labels={"hashtags": "Hashtag", "count": "Mentions"}
        )
        c2_left_col.plotly_chart(fig, use_container_width=True)
        # st.markdown("""---""")

    with c2_middle_col:
        # c2_middle_col.markdown("##### Sentiments")
        c2_middle_col.subheader("Sentiments")
        # st.markdown("""---""")
        sentiments_df = get_sentiments(twitter_data)
        fig = px.pie(
            sentiments_df, 
            values='Value', 
            names='Sentiments',
            hole=0.3,
            # textinfo="label+percent",
            color_discrete_map={'Negative':'red','Positive':'green'}
        )
        c2_middle_col.plotly_chart(fig, use_container_width=True)
        # st.markdown("""---""")

    with c2_right_col:
        # c2_right_col.markdown("##### Most Mentioned Characters")
        c2_right_col.subheader("Most Mentioned Characters")
        # st.markdown("""---""")
        fig = px.bar(
            twitter_meta_data["movie_characters"][:10], 
            x="count", 
            y="movie_characters", 
            labels={"movie_characters": "Movie Character", "count": "Mentions"}
        )
        c2_right_col.plotly_chart(fig, use_container_width=True)
        # st.markdown("""---""")

    space(2)

    c3 = st.container()    
    c3_left_col, c3_right_col = c2.columns([6, 3])
    with c3_left_col:
        # c3_left_col.markdown("##### Trend")  
        c3_left_col.subheader("Trend")  
        # st.markdown("""---""")
        # c3.dataframe(twitter_meta_data["daily_report"])
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
        # st.markdown("""---""")

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
        # c3_right_col.markdown("##### Tweets")  
        c3_right_col.subheader("Tweets")
        # st.markdown("""---""")
        c3_right_col.dataframe(twitter_data.sample(100)["tweet"])


    analysis_log("{}: Reload Completed {} tweet(s)\n".format(time.asctime(), twitter_data.shape[0]))
    
except:
    print(sys.exc_info())
    with open(base_path + "ll.txt", "w", newline="\n") as tmp:
        tmp.write(str(sys.exc_info()))