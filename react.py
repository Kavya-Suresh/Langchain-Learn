from dotenv import load_dotenv
import os
from operator import itemgetter
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
# from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch

load_dotenv()


def triple(num: float) -> float:
    """Triples a number."""
    return float(num) * 3

tools = [TavilySearch(max_results=3), triple]
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0).bind_tools(tools)