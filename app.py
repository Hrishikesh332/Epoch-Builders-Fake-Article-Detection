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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain_community.llms import OpenAI
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
from langchain import FAISS

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
import pickle

import tiktoken

import openai

import requests
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
# print(os.getenv('TWITTER_API_KEY'))

# TwitterTweetReader = download_loader("TwitterTweetReader")

page_element="""
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://img.freepik.com/free-vector/dark-gradient-background-with-copy-space_53876-99548.jpg?size=626&ext=jpg&ga=GA1.1.1448711260.1706990400&semt=ais");
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
    INFERENCE_URL_PROP = "https://api-inference.huggingface.co/models/valurank/distilroberta-propaganda-2class"
    INFERENCE_URL_HATE = "https://api-inference.huggingface.co/models/IMSyPP/hate_speech_en"


    headers = {"Authorization": os.getenv("Token")}

    def query_summ(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    def queryprop(payload):
        response = requests.post(INFERENCE_URL_PROP, headers=headers, json=payload)
        return response.json()

    def query_hate(payload):
        response = requests.post(INFERENCE_URL_HATE, headers=headers, json=payload)
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
        st.subheader("Summarization -")
        output = query_summ({
	            "inputs": c,
                })
        # summ=output[0]['summary_text']
        
        st.write(output[0]['summary_text'])
        f1=sent(c)
        st.subheader("Sentiment Analysis -")
        st.plotly_chart(f1)


        st.subheader("Hate Speech Detection -")
        output_hate=query_hate({
        "inputs": c,})
        # print(output[0][0])
        result = {}
        # st.write(output_hate[0])
        for data in output_hate[0]:
            if data['label'] == "LABEL_0":
                result["ACCEPTABLE"] = data['score']
            elif data['label'] == "LABEL_1":
                result["INAPPROAPRIATE"] = data['score']
            elif data['label'] == "LABEL_2":
                result["OFFENSIVE"] = data['score']
            elif data['label'] == "LABEL_3":
                result["VIOLENT"] = data['score']
        labels = list(result.keys())
        values = list(result.values())
        result_dict = dict(zip(labels, values))
        json_str = json.dumps(result_dict)
        data_parsed = json.loads(json_str)


        fig = go.Figure(go.Pie(labels=list(data_parsed.keys()), values=list(data_parsed.values())))
        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(family="Courier New, monospace", size=12, color="#7f7f7f"))
        )
        st.plotly_chart(fig, use_container_width=True)
        output_prop = queryprop({
        "inputs":  c,
        })
        
        yes = output_prop[0][0]['score']
        no = 1 - yes
        labels_prop = ["Propoganda", "Non-Propoganda"]
        values_prop = [yes, no]
        result_dict = dict(zip(labels_prop, values_prop))
        json_str = json.dumps(result_dict)
        data_parsed = json.loads(json_str)
        fig3 = go.Figure(go.Bar(x=labels_prop, y=values_prop))
        fig3.update_layout(
            autosize=False,
            width=500,
            height=500,
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(family="Courier New, monospace", size=12, color="#7f7f7f"))
        )

        st.write("Propaganda Detection -")

        st.plotly_chart(fig3, use_container_width=True)

        






persist_directory = 'Model/db'
model_name = "sentence-transformers/all-mpnet-base-v2"
instructor_embeddings = HuggingFaceEmbeddings(
model_name=model_name,
model_kwargs={'device': 'cpu'},
encode_kwargs={'normalize_embeddings': False}
)

embedding = instructor_embeddings
import textwrap
llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    st.write(wrap_text_preserve_newlines(llm_response['result']))
    st.write('\n\nSources:')
    for source in llm_response["source_documents"]:
        st.write(source.metadata['source'])
        
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 1})
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(api_key=os.getenv('OPENAI_API_KEY')),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

def process_text(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

if selected=="Chat":
    # ques=st.text_input("Ask any question and Confirm Fake News")
    # submit_ques=st.button("Click Here")
    # if submit_ques:
        # query = ques
        # llm_response = qa_chain("From the context, Do verify it and provide the correct information related to the same - "+ query)
        # process_llm_response(llm_response)
        pdf="Model/news.pdf"
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Create the knowledge base object
            knowledgeBase = process_text(text)
            query = st.text_input('Ask a question to the PDF')
            cancel_button = st.button('Cancel')
            
            if cancel_button:
                st.stop()
            
            if query:
                docs = knowledgeBase.similarity_search(query)
                chain = load_qa_chain(llm= OpenAI(api_key=os.getenv('OPENAI_API_KEY')), chain_type='stuff')
                
                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)
                    print(cost)
                    
                st.write(response)



 
    # load_dotenv()
    # print(os.getenv('OPENAI_API_KEY'))
 

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