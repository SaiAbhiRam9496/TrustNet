import os
from typing import List, Dict, Any, Optional
import httpx
from app.core.config import settings


class TavilySearchProvider:
    """
    Search provider interfacing with Tavily AI API for real-time web retrieval.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            # Fallback warning if API key is not configured
            return []

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(self.base_url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                results = []
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                        "score": item.get("score", 0.0)
                    })
                return results
        except Exception as e:
            # Return empty list on network error to allow fallback scraper handling
            return []
