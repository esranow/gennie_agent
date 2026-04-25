import streamlit as st, time, os
from retrieval import retrieve
from search import web_search
from routing import get_backend
from generation import generate
from ingestion import build_index
st.set_page_config(page_title="Spark", layout="wide")
st.title("⚡ Spark RAG")
with st.sidebar:
    st.header("Settings")
    up = st.file_uploader("Upload", accept_multiple_files=True)
    if up:
        os.makedirs("./data", exist_ok=True)
        for f in up:
            with open(os.path.join("./data", f.name), "wb") as w: w.write(f.getbuffer())
        st.success("Uploaded")
    if st.button("Build Index"):
        with st.spinner("Wait..."): build_index(); st.success("Done")
q = st.text_input("Ask:")
if q:
    t0 = time.time()
    with st.status("Thinking..."):
        cks, suf = retrieve(q)
        ws = [] if suf else web_search(q)
        be = get_backend()
        res = generate(q, cks, ws, be)
    st.metric("Latency", f"{(time.time()-t0)*1000:.0f}ms")
    st.markdown(res.get("final", ""))
    with st.expander("Debug"): st.write(res)
