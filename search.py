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


from tavily import TavilyClient

def web_search(query: str) -> list[str]:
    client = TavilyClient(api_key=get_secret("TAVILY_API_KEY"))
    res = client.search(query=query, search_depth="basic", max_results=2)
    return [r["content"] for r in res.get("results", [])]
