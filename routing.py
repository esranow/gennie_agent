import os, requests, streamlit as st
def get_sec(k):
    try: return st.secrets[k]
    except: return os.getenv(k)
def get_backend():
    try:
        requests.head("https://www.google.com", timeout=2)
        return "gemini"
    except: return "gemini"
def run_local(p):
    return "Local mode disabled. Please install transformers/torch for local inference."
