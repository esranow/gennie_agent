import config, re, tiktoken
from llama_index.core import Settings
config.setup_llm()
def generate(q, cks, ws, be):
    md = []
    for c in cks + ws:
        if c not in md: md.append(c)
    en = tiktoken.get_encoding("cl100k_base")
    tm, ct = [], 0
    for c in md:
        t = en.encode(c)
        if ct + len(t) > 3000:
            tm.append(en.decode(t[:3000-ct]))
            break
        tm.append(c); ct += len(t)
    ctx = "\n".join(tm)
    ms = "RAG+SEARCH" if ws else "RAG"
    sys = f"You are a QA system. You MUST respond in this format:\nMODE: {ms}\nTHINK: <reasoning>\nFINAL: <answer>\nCONFIDENCE: <high/medium/low>\n\nContext:\n{ctx}"
    rs = Settings.llm.complete(sys + "\nQuery: " + q).text
    m = re.search(r"MODE:\s*(.*?)\s*THINK:\s*(.*?)\s*FINAL:\s*(.*?)\s*CONFIDENCE:\s*(.*)", rs, re.S|re.I)
    if not m: return {"mode":ms, "think":[], "final":rs, "confidence":"low", "prompt_tokens":0, "completion_tokens":0}
    return {"mode":m.group(1).strip(),"think":m.group(2).strip().split("\n"),"final":m.group(3).strip(),"confidence":m.group(4).strip(),"prompt_tokens":len(en.encode(sys+q)),"completion_tokens":len(en.encode(rs))}
