from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


import streamlit as st
import os
from dotenv import load_dotenv

os.environ["OPEN_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true" # for langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system"," You are a University Chatbot. Please response to the queries"),
        ("user","Question:{question}")
    ]
)

# Streamlit framework

st.title("LangChain Demo with OpenAI API")
input_text=st.text_input("Search the queries you want")

# openAI LLM
llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()

chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
    

    
