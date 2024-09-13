from pymongo import MongoClient
from langchain.prompts import PromptTemplate
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
import google.generativeai as genai
from langchain_core.runnables import RunnablePassthrough
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")

# MongoDB Atlas connection
uri = f'mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@cluster0.5sbsz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
client = MongoClient(uri)

# Select the database and collection
db = client['Constitution']  
collection = db['policy']    

# Define the Atlas Vector Search Index name
ATLAS_VECTOR_SEARCH_INDEX_NAME = 'vector_index'

def get_vector_retriever():
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    # Define your vector search engine using MongoDB Atlas
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=[],  # no need to pass documents here if they are already in MongoDB
        embedding=embedding_model,
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )
    retriever = vector_search.as_retriever(search_type='similarity', search_kwargs={'k': 1})

    return retriever

st.title('Constitution of Nepal')


model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Based on the following context:\n{context}\nAnswer the question: in detail {question} if the answer is not in the context simply say i don't know"
)

retriever = get_vector_retriever()

rag_chain = (
   {
      "context": retriever,
      "question": RunnablePassthrough()
   }
   | prompt_template
   | model
   | StrOutputParser()
)
question = st.text_input('Enter Topic:')
if question:
    answer = rag_chain.invoke(question)
    st.write(f'Response: {answer}')



