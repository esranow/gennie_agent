import config, os
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_parse import LlamaParse
from llama_index.vector_stores.chroma import ChromaVectorStore
def build_index():
    co = config.get_chroma()
    vs = ChromaVectorStore(chroma_collection=co)
    sc = StorageContext.from_defaults(vector_store=vs)
    ps = LlamaParse(api_key=config.get_sec("LLAMA_KEY"), result_type="markdown")
    dd = Path("./data")
    docs = []
    if dd.exists():
        for f in dd.iterdir():
            if f.suffix.lower() in [".pdf",".docx",".txt"]: docs.extend(ps.load_data(str(f)))
    if not docs: return None
    return VectorStoreIndex.from_documents(docs, storage_context=sc)
if __name__ == "__main__": 
    config.setup_llm()
    build_index()
