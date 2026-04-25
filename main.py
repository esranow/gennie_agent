import json
import time
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from retrieval import retrieve
from search import web_search
from routing import get_backend
from generation import generate

app = FastAPI()

class QueryReq(BaseModel):
    query: str

@app.post("/query")
async def query_endpoint(req: QueryReq):
    start = time.time()
    query = req.query
    
    chunks, sufficient = retrieve(query)
    web_results = []
    if not sufficient:
        web_results = web_search(query)
        
    backend = get_backend()
    res = generate(query, chunks, web_results, backend)
    
    latency = (time.time() - start) * 1000
    mode = "RAG+SEARCH" if web_results else "RAG"
    
    log_data = {
        "mode": mode,
        "backend": backend,
        "latency_ms": latency,
        "prompt_tokens": res["prompt_tokens"],
        "completion_tokens": res["completion_tokens"]
    }
    print(json.dumps(log_data))
    
    return {
        "mode": res["mode"],
        "think": res["think"],
        "final": res["final"],
        "confidence": res["confidence"],
        "backend": backend,
        "latency_ms": latency
    }
