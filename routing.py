import os
import requests
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

_local_model = None
_local_tok = None

def get_backend() -> Literal["gemini", "local"]:
    try:
        requests.head("https://www.google.com", timeout=2)
        return "gemini"
    except (requests.ConnectionError, requests.Timeout):
        return "local"

def run_local(prompt: str) -> str:
    global _local_model, _local_tok
    if _local_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tk = os.getenv("HF_TOKEN")
        _local_tok = AutoTokenizer.from_pretrained("Srikri7/qwen3.5-2b-reasoning", token=tk)
        _local_model = AutoModelForCausalLM.from_pretrained(
            "Srikri7/qwen3.5-2b-reasoning",
            token=tk,
            torch_dtype="auto",
            device_map="auto",
        )
    inputs = _local_tok(prompt, return_tensors="pt").to(_local_model.device)
    outputs = _local_model.generate(**inputs, max_new_tokens=512, do_sample=False)
    out_text = _local_tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return out_text
