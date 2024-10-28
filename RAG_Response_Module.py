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


llm = ChatOpenAI(model="gpt-4o-mini")


def RAG_response(retriever, question):
  """LLM Component + RAG Response from the System Design of this project is implemened here.
      Inputs:
            retriever: The RAG retrirver system.
            question: User's query.
      Outputs:
            response: The RAG-enabled LLM's response to the user query given the relevant information from the knowledge base.
  """


  def rag_prompt(context, question):
      return f"""
  You are a RAG AI Agent. Your responsibility to is to answer a user query given in the "Question" section below, with a focus a set of documents that have been stored in the RAG system.
  The relevant content to the user query is given in the "Relevant content" section below.

  "Relevant Content":
  {context}

  "Question": {question}

  Answer the query based on the relevant content that is provided. The relevant content MUST be the centre of your answer, and you can add complementary content to it based on your knowledge.
  """

  # Define the RAG chain using RunnableMap and RunnableLambda
  rag_chain = RunnableMap(
      {
          "context": retriever,
          "question": RunnablePassthrough()
      }
  ) | RunnableLambda(lambda inputs: rag_prompt(inputs["context"], inputs["question"])) | llm | StrOutputParser()

  # Example query to the system
  response = rag_chain.invoke(question)
  return response
