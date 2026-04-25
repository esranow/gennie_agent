import config
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
def retrieve(q):
    try:
        co = config.get_chroma()
        if co.count() == 0: return [], False
    except: return [], False
    vs = ChromaVectorStore(chroma_collection=co)
    sc = StorageContext.from_defaults(vector_store=vs)
    idx = VectorStoreIndex.from_vector_store(vs, storage_context=sc)
    nds = idx.as_retriever(similarity_top_k=3).retrieve(q)
    cks = [n.node.get_content() for n in nds]
    qe = Settings.embed_model.get_text_embedding(q)
    ds = []
    for c in cks:
        ce = Settings.embed_model.get_text_embedding(c)
        dt = sum(a*b for a,b in zip(qe, ce))
        nq, nc = sum(a*a for a in qe)**0.5, sum(a*a for a in ce)**0.5
        ds.append(1.0 - (dt/(nq*nc) if nq*nc>0 else 0))
    return cks, (sum(ds)/len(ds) <= 0.45 if ds else False)
