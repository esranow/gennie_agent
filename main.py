import json, time, config
from fastapi import FastAPI
from pydantic import BaseModel
from retrieval import retrieve
from search import web_search
from generation import generate
config.setup_llm()
app = FastAPI()
class QReq(BaseModel): q: str
@app.post("/query")
async def q_ep(req: QReq):
    st_t = time.time()
    cks, suf = retrieve(req.q)
    ws = [] if suf else web_search(req.q)
    res = generate(req.q, cks, ws, "gemini")
    lt = (time.time() - st_t) * 1000
    return {**res, "latency":lt}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
