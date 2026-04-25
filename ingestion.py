import os
import streamlit as st
from dotenv import load_dotenv

# Load for local dev
load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)


from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_parse import LlamaParse
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from pathlib import Path

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="rag_store",
    metadata={"hnsw:space": "cosine"},
)
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64, paragraph_separator="\n\n")

Settings.llm = Gemini(
    model="models/gemini-2.5-pro",
    api_key=get_secret("GEMINI_KEY"),
    temperature=0.1,
    transport="rest",
)
Settings.embed_model = GeminiEmbedding(
    model_name="models/gemini-embedding-001",
    api_key=get_secret("GEMINI_KEY"),
    transport="rest",
)
Settings.transformations = [splitter]

def build_index() -> VectorStoreIndex:
    parser = LlamaParse(api_key=get_secret("LLAMA_KEY"), result_type="markdown", verbose=True)
    data_dir = Path("./data")
    raw_docs = []
    if data_dir.exists():
        for file in data_dir.iterdir():
            if file.suffix.lower() in [".pdf", ".docx", ".pptx", ".txt"]:
                raw_docs.extend(parser.load_data(str(file)))
    
    index = VectorStoreIndex.from_documents(raw_docs, storage_context=storage_context)
    return index

if __name__ == "__main__":
    build_index()
