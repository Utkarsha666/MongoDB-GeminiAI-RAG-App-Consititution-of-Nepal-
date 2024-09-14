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
APP_PASSWORD = os.getenv("APP_PASSWORD")
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def authenticate_user(app_password):
    if app_password != APP_PASSWORD:
        return False
    return True

def connect_mongoDB():
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")

    # MongoDB Atlas connection
    uri = f'mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@cluster0.5sbsz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    client = MongoClient(uri)

    # Select the database and collection
    db = client['Constitution']  
    collection = db['policy']    
    return collection


def get_vector_retriever(collection):
    # Define the Atlas Vector Search Index name
    ATLAS_VECTOR_SEARCH_INDEX_NAME = 'vector_index'
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    # Define your vector search engine using MongoDB Atlas
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=[],  # no need to pass documents here if they are already in MongoDB
        embedding=embedding_model,
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )
    retriever = vector_search.as_retriever(search_type='similarity', search_kwargs={'k': 2})

    return retriever

def run():

    st.title('Constitution of Nepal')


    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    template = """
        You are a legal assistant with expertise in the constitutional law of Nepal.\n\nYour task is to answer the following based only on the provided context: {context}\n\n
        Answer the question: {question} in as much detail as possible, but only using information from the context provided.\n\n
        When answering question:
        1. Provide precise and accurate information based strictly on the retrieved documents.
        2. Cite relevant articles or sections of the Constitution of Nepal.
        3. Do not provide information that is not present in the retrieved documents, and avoid making any assumptions.
        4. If the query is unclear or requires additional context, ask clarifying questions before providing an answer.
    """


    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    database = connect_mongoDB()
    retriever = get_vector_retriever(database)

    rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | model
    | StrOutputParser()
    )
    question = st.text_input('Ask Question about Constitution')
 
    if st.button("Submit Question"):
        if question:  
            answer = rag_chain.invoke(question)
            st.write(answer)
        else:
            st.error("Please enter a question")

placeholder = st.empty()

with placeholder.container():
    app_password = st.text_input("Enter password:", type="password")

if st.button("Submit"):
    if app_password:  
        if not authenticate_user(app_password):
            st.error("Invalid Password!!! Access Denied")
            st.stop()
    else:
        placeholder.empty()
          

if __name__ == "__main__":
    if (app_password == APP_PASSWORD):
        run()



