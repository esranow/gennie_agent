import os
from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient
from langsmith import traceable

@traceable
def web_search(query: str) -> list[str]:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    res = client.search(query=query, search_depth="basic", max_results=2)
    return [r["content"] for r in res.get("results", [])]
