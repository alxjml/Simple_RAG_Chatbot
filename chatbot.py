import time
import os
import re
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import getpass
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.runnables import RunnableMap, RunnableLambda
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
import bs4
import fitz
import streamlit as st

from file_parser import pdf_to_langchain_docs
from RAG_Response_Module import RAG_response
from RAG_Retriever_Module import RAG_retriever




def get_documents():
    pdf_file = "Thesis.pdf"
    documents = pdf_to_langchain_docs(pdf_path=pdf_file)
    return documents



def initialize_retriever(pinecone_api_key, chunk_size, chunk_overlap, dimension):
    # Load documents and initialize the retriever only after the API key is provided
    docs = get_documents()
    retriever = RAG_retriever(pinecone_api_key, docs, chunk_size, chunk_overlap, dimension)
    return retriever

def main():
    st.title("Chat with Video")
    st.write("Welcome to the Chat with Video chatbot! Type your questions below.")

    # Sidebar inputs
    pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
    chunk_size = st.sidebar.number_input("Chunk Size", min_value=1, value=500)
    chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, value=50)
    dimension = st.sidebar.number_input("Dimension", min_value=1, value=768)

    # Button to initialize the chat
    if st.sidebar.button("Initialize Chat"):
        if pinecone_api_key:
            # Initialize retriever and conversation history
            st.session_state.retriever = initialize_retriever(pinecone_api_key, chunk_size, chunk_overlap, dimension)
            st.session_state.conversation = []  # Initialize conversation history
            st.success("Chat initialized! You can start asking questions.")
        else:
            st.warning("Please enter your Pinecone API key to initialize the chat.")

    # Check if the retriever is initialized
    if "retriever" in st.session_state:


        for message in st.session_state.conversation:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.conversation.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            response = RAG_response(st.session_state.retriever, prompt)
            st.session_state.conversation.append({"role": "bot", "content": response})
            with st.chat_message("bot"):
                st.markdown(response)


    else:
        st.info("Please initialize the chat by entering the required parameters in the sidebar.")

if __name__ == "__main__":
    main()