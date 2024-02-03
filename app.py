import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import requests
import os
from goose3 import Goose
from wordcloud import WordCloud, STOPWORDS
import json
import io
from PIL import Image
import base64
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio


from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from dotenv import load_dotenv
from langchain_community.document_loaders import TwitterTweetLoader

from llama_index import SimpleDirectoryReader
from llama_index import GPTVectorStoreIndex
from langchain_community.document_loaders import TwitterTweetLoader
import llama_index


import requests

	

# print(os.getenv('TWITTER_API_KEY'))

# TwitterTweetReader = download_loader("TwitterTweetReader")

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


selected = option_menu(
    menu_title=None,
    options=["Main","Chat"],
    menu_icon=['house', 'chat-dots'],
    orientation="horizontal")


st.markdown(page_element, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: white';>Samachar AI 💬</h1>", unsafe_allow_html=True)
st.markdown("---")

if selected=="Main":
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": os.getenv('Token')}

    def query_summ(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    link=st.text_input("Put the Article link below to check -")
    check=st.button("Check and Analyze")
    if check:
        def clean(link):
            url = link
            goose = Goose()
            articles = goose.extract(url)
            sentence1 = articles.cleaned_text
            return sentence1
        def sent(sentence1):
            senti=[]
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

            return fig
        c=clean(link)
        f1=sent(c)
        st.plotly_chart(f1)
        output = query_summ({
	            "inputs": c,
                })
        st.subheader("Summarization -")
        st.write(output[0]['summary_text'])


if selected=="Chat":
    pass
    # load_dotenv()
    # print(os.getenv('OPENAI_API_KEY'))
    # os.getenv('OPENAI_API_KEY')

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # if prompt := st.chat_input("Welcome to Llama Press, How may I help you?"):
    #     with st.chat_message("user"):
    #         st.markdown(prompt)
    #     st.session_state.messages.append({"role": "user", "content": prompt})

    # def chat(p):
    #     query = p
    #     # reader = TwitterTweetReader(os.getenv('BEARER_TOKEN'))
    #     # documents = reader.load_data(["ANI"])
    #     # documents1 = reader.load_data(["ZeeNews"])
    #     # documents2 = reader.load_data(["TV9Bharatvarsh"])
    #     # documents3 = reader.load_data(["Republic_Bharat"])
    #     # documents4 = reader.load_data(["AajTak"])
    #     loader = TwitterTweetLoader.from_secrets(
       
    #         access_token=os.getenv('ACCESS_TOKEN'),
    #         access_token_secret=os.getenv('ACCESS_TOKEN_SECRET'),
    #         consumer_key=os.getenv('CONSUMER_API_KEY'),
    #         consumer_secret=os.getenv('CONSUMER_API_KEY_SECRET'),
    #         twitter_users=['elonmusk'],
    #         number_tweets=50,
    #     )
    #     documents = loader.load()
    #     st.write(documents[:5])
        
    #     # agent = llama_index.GPTVectorStoreIndex.from_documents(documents)
    #     # chat_engine = agent.as_chat_engine(verbose=True)
    #     # response = chat_engine.chat(query)
    #     # return response

    # result = chat(prompt)
    # response = f"Echo: {result}"
    # with st.chat_message("assistant"):
    #     st.markdown(response)
    # st.session_state.messages.append({"role": "assistant", "content": response})