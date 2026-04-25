import os, streamlit as st, chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
def get_sec(k):
    try: return st.secrets[k]
    except: return os.getenv(k)
cl = chromadb.PersistentClient(path="./chroma_db")
co = cl.get_or_create_collection(name="rag_store", metadata={"hnsw:space": "cosine"})
vs = ChromaVectorStore(chroma_collection=co)
sc = StorageContext.from_defaults(vector_store=vs)
sp = SentenceSplitter(chunk_size=512, chunk_overlap=64)
Settings.llm = Gemini(model="models/gemini-2.5-pro", api_key=get_sec("GEMINI_KEY"), transport="rest")
Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=get_sec("GEMINI_KEY"), transport="rest")
Settings.transformations = [sp]
def retrieve(q):
    idx = VectorStoreIndex.from_vector_store(vs, storage_context=sc)
    ret = idx.as_retriever(similarity_top_k=3)
    nds = ret.retrieve(q)
    cks = [n.node.get_content() for n in nds]
    qe = Settings.embed_model.get_text_embedding(q)
    ds = []
    for c in cks:
        ce = Settings.embed_model.get_text_embedding(c)
        dt = sum(a*b for a,b in zip(qe, ce))
        nq = sum(a*a for a in qe)**0.5
        nc = sum(a*a for a in ce)**0.5
        sim = dt/(nq*nc) if nq*nc>0 else 0
        ds.append(1.0-sim)
    md = sum(ds)/len(ds) if ds else 1.0
    return cks, md <= 0.45
