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
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

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

def retrieve(query: str) -> tuple[list[str], bool]:
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(query)
    
    chunks = [n.node.get_content() for n in nodes]
    
    q_emb = Settings.embed_model.get_text_embedding(query)
    dists = []
    for c in chunks:
        c_emb = Settings.embed_model.get_text_embedding(c)
        dot = sum(a * b for a, b in zip(q_emb, c_emb))
        norm_q = sum(a * a for a in q_emb) ** 0.5
        norm_c = sum(a * a for a in c_emb) ** 0.5
        cos_sim = dot / (norm_q * norm_c) if (norm_q * norm_c) > 0 else 0
        dists.append(1.0 - cos_sim)
        
    mean_dist = sum(dists) / len(dists) if dists else 1.0
    sufficient = mean_dist <= 0.45
    
    return chunks, sufficient
