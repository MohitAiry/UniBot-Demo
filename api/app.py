from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes

import uvicorn
import os

from dotenv import load_dotenv
load_dotenv()
from langchain_community.llms import Ollama 

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")


app=FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API Server"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model=ChatOpenAI()

##Ollama llama2
llm=Ollama(model="gemma:2b")

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me a poem about {topic} in 100 words")

add_routes(
    app,
    prompt1|model,
    path="/essay"
)

add_routes(
    app,
    prompt2|llm,
    path="/poem"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)