import os

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path="D:\\Code\\langchain-learn\\Langchain-Learn\\src\\langchain_learn\\mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=1000000,
    )
    document = loader.load()
    print(f"Created {len(document)} chunks")

    print("splitting ....")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"created {len(texts)} chunks of data")

    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("Ingesting...")

    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ.get("INDEX_NAME")
    )
    print("Ingestion completed.")

    pass
