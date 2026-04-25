import os, chromadb
try: import streamlit as st
except: st = None
def get_sec(k):
    if st and k in st.secrets: return st.secrets[k]
    return os.getenv(k)
@st.cache_resource
def get_chroma():
    cl = chromadb.PersistentClient(path="./chroma_db")
    return cl.get_or_create_collection(name="rag_store", metadata={"hnsw:space":"cosine"})
def setup_llm():
    from llama_index.core import Settings
    from llama_index.llms.gemini import Gemini
    from llama_index.embeddings.gemini import GeminiEmbedding
    from llama_index.core.node_parser import SentenceSplitter
    Settings.llm = Gemini(model="models/gemini-2.5-pro", api_key=get_sec("GEMINI_KEY"), transport="rest")
    Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=get_sec("GEMINI_KEY"), transport="rest")
    Settings.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=64)]
