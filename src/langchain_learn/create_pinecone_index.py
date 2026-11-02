import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# 1. Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "document-helper-index"

# 2. Create a classic vector index (Bypassing the Web UI Schema)
print(f"Creating classic vector index '{index_name}'...")
pc.create_index(
    name=index_name,
    dimension=768,  # Exact dimensions for Ollama's nomic-embed-text
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

print("Success! You can now run your main ingestion script.")
