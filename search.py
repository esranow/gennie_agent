import config
from tavily import TavilyClient
def web_search(q):
    c = TavilyClient(api_key=config.get_sec("TAVILY_API_KEY"))
    r = c.search(query=q, search_depth="basic", max_results=2)
    return [x["content"] for x in r.get("results", [])]
