import streamlit as st
import time
from retrieval import retrieve
from search import web_search
from routing import get_backend
from generation import generate
from ingestion import build_index

st.set_page_config(page_title="Spark RAG Pipeline", page_icon="⚡", layout="wide")

st.title("⚡ Spark Modular RAG Pipeline")

with st.sidebar:
    st.header("⚙️ Admin & Settings")
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=["pdf", "docx", "pptx", "txt"])
    if uploaded_files:
        import os
        os.makedirs("./data", exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join("./data", file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} files.")
        
    st.markdown("Use this to rebuild the local vector store index from the documents.")
    if st.button("Build/Update Index"):
        with st.spinner("Building index..."):
            build_index()
            st.success("Index built successfully!")
            
    st.markdown("---")
    st.markdown("""
    **Pipeline Features:**
    - 🔍 ChromaDB Vector Retrieval
    - 🌐 Fallback Web Search (Tavily)
    - 🧠 Dynamic Routing (Gemini/Local)
    - ⚡ Fast Generation
    """)

query = st.text_input("Enter your question:", placeholder="e.g. What is the attention mechanism?")

if st.button("Submit", type="primary") or query:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        start_time = time.time()
        
        with st.status("Processing query...", expanded=True) as status:
            st.write("🔍 Retrieving context from local ChromaDB...")
            chunks, sufficient = retrieve(query)
            st.write(f"Found {len(chunks)} relevant chunks. Context sufficient: `{sufficient}`")
            
            web_results = []
            if not sufficient:
                st.write("🌐 Context insufficient. Falling back to Tavily Web Search...")
                web_results = web_search(query)
                st.write(f"Retrieved {len(web_results)} web search results.")
                
            backend = get_backend()
            st.write(f"🧠 Routing to backend model: `{backend}`")
            
            st.write("✨ Generating final response...")
            res = generate(query, chunks, web_results, backend)
            
            status.update(label="Query processed successfully!", state="complete", expanded=False)
            
        latency = (time.time() - start_time) * 1000
        mode = "RAG+SEARCH" if web_results else "RAG"
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Backend Model", backend.upper())
        col2.metric("Pipeline Mode", mode)
        col3.metric("Latency", f"{latency:.2f} ms")
        col4.metric("Confidence", res.get("confidence", "N/A"))
        
        st.subheader("💡 Final Answer")
        st.markdown(res.get("final", ""))
        
        st.markdown("---")
        st.subheader("🛠️ Debug Information")
        with st.expander("🤔 Internal Thinking / Chain of Thought"):
            st.markdown(res.get("think", "No thinking process available."))
            
        with st.expander("📚 Retrieved Local Context"):
            if chunks:
                for i, chunk in enumerate(chunks):
                    st.info(f"**Chunk {i+1}:** {chunk}")
            else:
                st.warning("No local context retrieved.")
                
        with st.expander("🌐 Web Search Results"):
            if web_results:
                for i, res_web in enumerate(web_results):
                    st.info(f"**Web Result {i+1}:**\n\n{res_web}")
            else:
                st.write("No web search was performed.")
