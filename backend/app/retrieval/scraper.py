from typing import Optional, Dict, Any
from urllib.parse import urlparse
import trafilatura


def extract_article_content(url: str, raw_html: Optional[str] = None, fallback_snippet: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts article text and domain authority metadata from a URL.
    Uses fallback_snippet or trafilatura text extraction.
    """
    domain = urlparse(url).netloc.replace("www.", "")
    
    # Determine Domain Authority Tier
    domain_lower = domain.lower()
    if domain_lower.endswith((".gov", ".edu", ".mil")) or any(
        top_news in domain_lower for top_news in [
            "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
            "washingtonpost.com", "theguardian.com", "nature.com", "sciencedirect.com"
        ]
    ):
        authority_tier = "High"
    elif any(
        mid_news in domain_lower for mid_news in [
            "cnn.com", "nbcnews.com", "bloomberg.com", "forbes.com", "wikipedia.org", "time.com"
        ]
    ):
        authority_tier = "Medium"
    else:
        authority_tier = "Low"

    # If a clean snippet is already provided (e.g. from Tavily API), return immediately
    if fallback_snippet and len(fallback_snippet.strip()) > 20:
        return {
            "url": url,
            "domain": domain,
            "authority_tier": authority_tier,
            "title": "",
            "text": fallback_snippet.strip()
        }

    # Fetch HTML if raw HTML was passed or attempt fast extract
    html = raw_html
    if not html:
        return {
            "url": url,
            "domain": domain,
            "authority_tier": authority_tier,
            "title": "",
            "text": fallback_snippet or ""
        }

    extracted_text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False
    ) or fallback_snippet or ""

    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata and metadata.title else ""

    return {
        "url": url,
        "domain": domain,
        "authority_tier": authority_tier,
        "title": title,
        "text": extracted_text
    }
