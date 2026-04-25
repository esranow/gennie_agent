import os
import re
import tiktoken
import streamlit as st
from dotenv import load_dotenv

# Load for local dev
load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

from langsmith import traceable
from routing import run_local

from llama_index.core import Settings
from llama_index.llms.gemini import Gemini

Settings.llm = Gemini(
    model="models/gemini-2.5-pro",
    api_key=get_secret("GEMINI_KEY"),
    temperature=0.1,
    transport="rest",
)

@traceable
def generate(query: str, chunks: list[str], web_results: list[str], backend: str) -> dict:
    from langsmith.run_helpers import get_current_run
    run = get_current_run()
    if run:
        tags = ["offline"] if backend == "local" else []
        tags.append("rag+search" if web_results else "rag-only")
        run.add_tags(tags)
        
    merged = []
    for c in chunks + web_results:
        if c not in merged:
            merged.append(c)
            
    enc = tiktoken.get_encoding("cl100k_base")
    trunc_merged = []
    curr_tokens = 0
    for c in merged:
        t = enc.encode(c)
        if curr_tokens + len(t) > 3000:
            rem = 3000 - curr_tokens
            trunc_merged.append(enc.decode(t[:rem]))
            break
        trunc_merged.append(c)
        curr_tokens += len(t)
        
    context = "\n".join(trunc_merged)
    mode_str = "RAG+SEARCH" if web_results else "RAG"
    sys_prompt = f"""You are a strict retrieval-based QA system with adaptive reasoning.

Use ONLY the provided CONTEXT and WEB results.
NEVER use prior or general knowledge.

PROCESS:
1. Check available information in CONTEXT and WEB.
2. Determine:
   * Sufficiency: (sufficient / partial / insufficient)
   * Complexity: (simple / complex)
3. If sufficient + simple → answer directly.
4. If sufficient + complex → reason step-by-step using ONLY provided data.
5. If partial:
   * If WEB is available → use it to complete missing info, then reason.
   * Else → return "INSUFFICIENT DATA"
6. If insufficient → return "INSUFFICIENT DATA"
7. NEVER infer or assume missing facts.

Respond EXACTLY in this format with no extra text:
MODE: {mode_str}
THINK:
* Available info summary
* Sufficiency: (sufficient / partial / insufficient)
* Complexity: (simple / complex)
* Missing info (if any)
* Reasoning steps (only if complex)
FINAL:
[Your answer using ONLY provided data, or "INSUFFICIENT DATA"]
CONFIDENCE:
[high, medium, or low]

Context:
{context}
"""
    prompt_tokens = len(enc.encode(sys_prompt + "\nQuery: " + query))
    
    if backend == "gemini":
        resp = Settings.llm.complete(sys_prompt + "\nQuery: " + query).text
    else:
        resp = run_local(sys_prompt + "\nQuery: " + query)
        resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
        
    completion_tokens = len(enc.encode(resp))
    
    m = re.search(r"MODE:\s*(RAG(?:\+SEARCH)?)\s*THINK:\s*(.*?)\s*FINAL:\s*(.*?)\s*CONFIDENCE:\s*(low|medium|high)", resp, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError(f"Output format not matched: {resp}")
        
    steps = [s.strip().lstrip("- *") for s in m.group(2).strip().split("\n") if s.strip()]
    return {
        "mode": m.group(1).strip().upper(),
        "think": steps,
        "final": m.group(3).strip(),
        "confidence": m.group(4).strip().lower(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }
