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





def RAG_retriever(pinecone_api_key, docs, chunk_size, chunk_overlap, dimension):
    """ The RAG Retriever component + RAG Vector DB component of the System Design of this project is implemented here.
        Inputs:
              pinecone_api_key: Since Pinecone Vectore DB is used, a pinecone api key is required.
              docs: The documents to be added to the knoweldge base.
    """
      # Split the documents into smaller chunks while preserving metadata
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)

    # Initialize Pinecone
    pc = Pinecone(api_key = pinecone_api_key)

    index_name = "langchain-index"  # change if desired

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name in existing_indexes:
        pc.delete_index(index_name)

    # if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())

    # Add the split documents to the vector store
    vector_store.add_documents(documents=splits)

    # Use the vector store as a retriever
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 3, "lambda_mult": 0.5})
    return retriever