import os, streamlit as st, chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_parse import LlamaParse
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
def get_sec(k):
    try: return st.secrets[k]
    except: return os.getenv(k)
cl = chromadb.PersistentClient(path="./chroma_db")
co = cl.get_or_create_collection(name="rag_store", metadata={"hnsw:space":"cosine"})
vs = ChromaVectorStore(chroma_collection=co)
sc = StorageContext.from_defaults(vector_store=vs)
sp = SentenceSplitter(chunk_size=512, chunk_overlap=64)
Settings.llm = Gemini(model="models/gemini-2.5-pro", api_key=get_sec("GEMINI_KEY"), transport="rest")
Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=get_sec("GEMINI_KEY"), transport="rest")
Settings.transformations = [sp]
def build_index():
    ps = LlamaParse(api_key=get_sec("LLAMA_KEY"), result_type="markdown")
    dd = Path("./data")
    docs = []
    if dd.exists():
        for f in dd.iterdir():
            if f.suffix.lower() in [".pdf", ".docx", ".txt"]: docs.extend(ps.load_data(str(f)))
    return VectorStoreIndex.from_documents(docs, storage_context=sc)
if __name__ == "__main__": build_index()
