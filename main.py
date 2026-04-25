import json, time, os, streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from retrieval import retrieve
from search import web_search
from routing import get_backend
from generation import generate
app = FastAPI()
class QReq(BaseModel): q: str
@app.post("/query")
async def q_ep(req: QReq):
    st_t = time.time()
    q = req.q
    cks, suf = retrieve(q)
    ws = [] if suf else web_search(q)
    be = get_backend()
    res = generate(q, cks, ws, be)
    lt = (time.time() - st_t) * 1000
    print(json.dumps({"mode":res["mode"], "latency":lt}))
    return {**res, "latency":lt}
