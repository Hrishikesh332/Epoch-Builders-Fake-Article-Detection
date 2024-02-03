import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import requests
from goose3 import Goose
from wordcloud import WordCloud, STOPWORDS
import json
from PIL import Image
import base64
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging


page_element="""
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://cdn.wallpapersafari.com/88/75/cLUQqJ.jpg");
background-size: cover;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"]> div:first-child{
background-image: url("https://mcdn.wallpapersafari.com/medium/89/87/X7GDE5.jpg");
background-size: cover;
}
</style>

"""

st.markdown(page_element, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: white';>Samachar AI 💬</h1>", unsafe_allow_html=True)
st.markdown("---")

link=st.text_input("Put the Article link below to check -")
check=st.button("Check and Analyze")
if check:

    def sent(link):
        senti=[]
        url = link
        goose = Goose()
        articles = goose.extract(url)
        sentence1 = articles.cleaned_text
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores([sentence1])
        print(sentiment_dict['neg']*100, "% Negative")
        print(sentiment_dict['pos']*100, "% Positive")
        print("Review Overall Analysis", end = " ") 
        if sentiment_dict['compound'] >= 0.05 :
            senti.append("Positive")
        elif sentiment_dict['compound'] <= -0.05 :
            senti.append("Negative")
        else :
            senti.append("Neutral")
        
        labels = list(sentiment_dict.keys())
        values = list(sentiment_dict.values())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            line=dict(color='darkblue'),
            marker=dict(size=10, color='royalblue', symbol='circle-open')
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title={
                'text': "Sentiment Analysis",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            font=dict(family="Courier New, monospace", size=12, color="#7f7f7f")
        )

        st.plotly_chart(fig)

    sent(link)
