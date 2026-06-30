"""
Web search tool, using Tavily (https://tavily.com) - the standard choice
for LLM-agent-integrated search: clean, pre-summarized JSON results (not
raw HTML to parse), built specifically for this use case, with a free
tier sufficient for portfolio/demo use.

Requires TAVILY_API_KEY in .env. Get one at https://tavily.com (free tier
available). This is a SEPARATE API key from your OpenAI key - a different
service entirely.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


class WebSearchError(Exception):
    """Raised when the web search request fails (missing key, network
    error, bad response) - kept as a distinct exception type so callers
    (the router/orchestrator) can catch this specifically and degrade
    gracefully (e.g. proceed with RAG/SQL results alone) rather than
    crashing the whole combined-answer flow over a search outage."""


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily. Returns a list of
    {"title": ..., "url": ..., "content": ...} dicts - "content" is
    Tavily's own short extracted summary of the page, not the full raw
    page text.

    Raises WebSearchError (not a generic exception) on any failure, so
    the caller can distinguish "search is unavailable right now" from a
    genuine bug elsewhere in the pipeline.
    """
    if not TAVILY_API_KEY:
        raise WebSearchError(
            "TAVILY_API_KEY not set in .env - web search is unavailable. "
            "Get a free key at https://tavily.com and add it to your .env file."
        )

    try:
        response = requests.post(
            TAVILY_API_URL,
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise WebSearchError(f"Web search request failed: {e}") from e

    data = response.json()
    results = data.get("results", [])
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in results
    ]


def format_search_results(results: list[dict]) -> str:
    """Render search results as a citable text block for an LLM prompt -
    same pattern as how retrieved chunks are formatted elsewhere in this
    project (source label + content), for consistency."""
    if not results:
        return "No web search results found."
    blocks = []
    for r in results:
        blocks.append(f"[Source: {r['title']} | {r['url']}]\n{r['content']}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    query = input("Enter a web search query: ").strip()
    try:
        results = web_search(query)
        print(f"\n{len(results)} results:\n")
        print(format_search_results(results))
    except WebSearchError as e:
        print(f"Search failed: {e}")
