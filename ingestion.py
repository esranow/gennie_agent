import config, os
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.parse.llamaparse import LlamaParse
from llama_index.vector_stores.chroma import ChromaVectorStore
config.setup_llm()
vs = ChromaVectorStore(chroma_collection=config.co)
sc = StorageContext.from_defaults(vector_store=vs)
def build_index():
    ps = LlamaParse(api_key=config.get_sec("LLAMA_KEY"), result_type="markdown")
    dd = Path("./data")
    docs = []
    if dd.exists():
        for f in dd.iterdir():
            if f.suffix.lower() in [".pdf",".docx",".txt"]: docs.extend(ps.load_data(str(f)))
    if not docs: return None
    return VectorStoreIndex.from_documents(docs, storage_context=sc)
if __name__ == "__main__": build_index()
