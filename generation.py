import os, re, tiktoken, streamlit as st
from routing import run_local
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
def get_sec(k):
    try: return st.secrets[k]
    except: return os.getenv(k)
Settings.llm = Gemini(model="models/gemini-2.5-pro", api_key=get_sec("GEMINI_KEY"), transport="rest")
def generate(q, cks, ws, be):
    md = []
    for c in cks + ws:
        if c not in md: md.append(c)
    en = tiktoken.get_encoding("cl100k_base")
    tm = []
    ct = 0
    for c in md:
        t = en.encode(c)
        if ct + len(t) > 3000:
            tm.append(en.decode(t[:3000-ct]))
            break
        tm.append(c)
        ct += len(t)
    ctx = "\n".join(tm)
    ms = "RAG+SEARCH" if ws else "RAG"
    sys = f"""MODE: {ms}\nTHINK:\nFINAL:\nCONFIDENCE:\nContext:\n{ctx}"""
    if be == "gemini":
        rs = Settings.llm.complete(sys + "\nQuery: " + q).text
    else:
        rs = run_local(sys + "\nQuery: " + q)
    m = re.search(r"MODE:\s*(.*?)\s*THINK:\s*(.*?)\s*FINAL:\s*(.*?)\s*CONFIDENCE:\s*(.*)", rs, re.S|re.I)
    if not m: return {"mode":ms, "think":[], "final":rs, "confidence":"low", "prompt_tokens":0, "completion_tokens":0}
    return {
        "mode": m.group(1).strip(),
        "think": m.group(2).strip().split("\n"),
        "final": m.group(3).strip(),
        "confidence": m.group(4).strip(),
        "prompt_tokens": len(en.encode(sys+q)),
        "completion_tokens": len(en.encode(rs))
    }
