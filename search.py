import os, streamlit as st
from tavily import TavilyClient
def get_sec(k):
    try: return st.secrets[k]
    except: return os.getenv(k)
def web_search(q):
    c = TavilyClient(api_key=get_sec("TAVILY_API_KEY"))
    r = c.search(query=q, search_depth="basic", max_results=2)
    return [x["content"] for x in r.get("results", [])]
