import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)
print("Loaded .env:", env_path.exists())
print("Pinecone key available:", bool(os.getenv("PINECONE_API_KEY")))


def format_docs(docs):
    """Format the retrieved documents into a string for the prompt."""
    return "\n\n".join([doc.page_content for doc in docs])


print("Initializing components...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen3:latest", temperature=0)

vector_store = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}
    
    Question: {question}
    
    Provide a detailed answer:"""
)


if __name__ == "__main__":
    print("Retrieving...")

    query = "What is the pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "*" * 50 + "\nImplementing 0: Raw LLM - No RAG...\n" + "*" * 50 + "\n")
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(f"Raw LLM Result: {result_raw.content}")

    # ========================================================================
    # Option 1: With RAG - No LCEL
    # ========================================================================
    print(
        "\n" + "*" * 50 + "\nImplementing 1: With RAG - No LCEL...\n" + "*" * 50 + "\n"
    )
    """Create a simple retrieval chain without LCEL"""
    docs = retriever.invoke(query)
    context = format_docs(docs)
    message = prompt_template.format_prompt(context=context, question=query)
    result_rag = llm.invoke(message)
    print(f"RAG Result: {result_rag.content}")

    # ========================================================================
    # Option 1: With RAG - LCEL
    # ========================================================================
    print("\n" + "*" * 50 + "\nImplementing 1: With RAG - LCEL...\n" + "*" * 50 + "\n")
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    result_rag_lcel = retrieval_chain.invoke({"question": query})
    print(f"RAG Result: {result_rag_lcel}")
    pass
