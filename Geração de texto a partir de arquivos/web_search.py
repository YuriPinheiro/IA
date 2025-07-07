from duckduckgo_search import DDGS

def search_web(query):
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return [r["body"] for r in results][:3]
