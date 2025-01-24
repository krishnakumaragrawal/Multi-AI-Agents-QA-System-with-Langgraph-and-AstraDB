import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from typing import List
from typing_extensions import TypedDict
from pprint import pprint
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize components
api_key = os.getenv('GROQ_API_KEY')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vector_store = AstraDBVectorStore(
    collection_name='langgraph_astra',
    embedding=embeddings,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT
)
astra_vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)
retriever = vector_store.as_retriever()

# Initialize Groq LLM
llm = ChatGroq(api_key=api_key, model_name="Gemma2-9b-It")

# Define the Router
class RouteQuery(BaseModel):
    datasource: Literal['vectorstore', 'wiki_search', 'arxiv_search'] = Field(
        ..., description="Given a user question, choose to route it to Wikipedia, a vectorstore, or Arxiv."
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

# Define the prompt for routing
system = """
You are an expert at routing a user question to a vectorstore, Wikipedia, or Arxiv.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search or arxiv-search.
"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])
question_route = route_prompt | structured_llm_router

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Define the Graph
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def retrieve(state):
    print("---RETRIEVE---")
    question = state['question']
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):
    print("---WIKIPEDIA SEARCH---")
    question = state['question']
    docs = wiki.invoke({"query": question})
    wiki_results = Document(page_content=docs)
    return {"documents": wiki_results, "question": question}

def arxiv_search(state):
    print("---ARXIV SEARCH---")
    question = state['question']
    docs = arxiv.invoke({"query": question})
    arxiv_results = Document(page_content=docs)
    return {"documents": arxiv_results, "question": question}

def route_question(state):
    print("---ROUTE QUESTION---")
    question = state['question']
    source = question_route.invoke({"question": question})
    if source.datasource == 'wiki_search':
        print("---Route Question to Wiki Search---")
        return "wiki_search"
    elif source.datasource == 'vectorstore':
        print("---Route Question to RAG---")
        return "vectorstore"
    elif source.datasource == 'arxiv_search':
        print("---Route Question to Arxiv Search---")
        return "arxiv_search"

workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("arxiv_search", arxiv_search)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
        "arxiv_search": "arxiv_search",
    },
)

workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
workflow.add_edge("arxiv_search", END)

app = workflow.compile()

# Streamlit App
st.title("Multi-Agent QA System")
st.write("Ask a question, and the system will retrieve answers from either the vector store, Wikipedia, or Arxiv.")

# Input question
question = st.text_input("Enter your question:")

if question:
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            st.write(f"Node '{key}':")
            st.write(value['documents'])