"""Web search for news-pulse and other skills.

Primary: Brave Search API (richer metadata, thumbnails, freshness controls).
Fallback: DuckDuckGo (free, no API key).
Optional: Brave AI Grounding for per-article summarization.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger("memoryos.agents.search")

_BRAVE_NEWS_URL = "https://api.search.brave.com/res/v1/news/search"
_BRAVE_WEB_URL = "https://api.search.brave.com/res/v1/web/search"

_FRESHNESS_MAP = {"d": "pd", "w": "pw", "m": "pm"}


def _brave_api_key() -> str | None:
    return os.environ.get("BRAVE_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY")


def _brave_ai_key() -> str | None:
    return os.environ.get("BRAVE_AI_API_KEY")


# ---------------------------------------------------------------------------
# Brave Search
# ---------------------------------------------------------------------------

def _search_brave_news(
    topic: str,
    keywords: list[str],
    max_results: int = 5,
    freshness: str = "pd",
) -> list[dict[str, Any]]:
    """Search Brave News API. Returns enriched result dicts."""
    api_key = _brave_api_key()
    if not api_key:
        return []

    query = " ".join(keywords)
    params: dict[str, Any] = {
        "q": query,
        "count": min(max_results, 20),
        "freshness": freshness,
        "extra_snippets": True,
    }
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }

    try:
        resp = requests.get(_BRAVE_NEWS_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Brave news search failed for '%s': %s", topic, exc)
        return []

    results: list[dict[str, Any]] = []
    for item in data.get("results", []):
        thumb = item.get("thumbnail", {})
        meta = item.get("meta_url", {})
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
            "date": item.get("age", ""),
            "source": meta.get("hostname", item.get("source", "")),
            "thumbnail_url": thumb.get("src", "") if isinstance(thumb, dict) else "",
            "favicon_url": meta.get("favicon", "") if isinstance(meta, dict) else "",
            "extra_snippets": item.get("extra_snippets", []),
        })

    logger.info("Brave news '%s': %d results", topic, len(results))
    return results


def _search_brave_web(
    topic: str,
    keywords: list[str],
    max_results: int = 5,
    freshness: str = "pd",
) -> list[dict[str, Any]]:
    """Fallback: Brave web search when news endpoint yields few results."""
    api_key = _brave_api_key()
    if not api_key:
        return []

    query = f"{' '.join(keywords)} news {datetime.now().strftime('%Y')}"
    params: dict[str, Any] = {
        "q": query,
        "count": min(max_results, 20),
        "freshness": freshness,
    }
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }

    try:
        resp = requests.get(_BRAVE_WEB_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Brave web search failed for '%s': %s", topic, exc)
        return []

    results: list[dict[str, Any]] = []
    for item in data.get("web", {}).get("results", []):
        thumb = item.get("thumbnail", {})
        meta = item.get("meta_url", {})
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
            "date": item.get("age", ""),
            "source": meta.get("hostname", ""),
            "thumbnail_url": thumb.get("src", "") if isinstance(thumb, dict) else "",
            "favicon_url": meta.get("favicon", "") if isinstance(meta, dict) else "",
            "extra_snippets": item.get("extra_snippets", []),
        })

    logger.info("Brave web '%s': %d results", topic, len(results))
    return results


# ---------------------------------------------------------------------------
# Brave AI Grounding (optional per-article summarization)
# ---------------------------------------------------------------------------

def summarize_with_brave_ai(query: str, max_tokens: int = 300) -> str | None:
    """Use Brave AI Grounding to get a web-grounded summary for a query.

    Returns the summary text, or None if unavailable.
    """
    api_key = _brave_ai_key()
    if not api_key:
        return None

    headers = {
        "X-Subscription-Token": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": "brave-search",
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        resp = requests.post(
            "https://api.search.brave.com/res/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content")
    except Exception as exc:
        logger.warning("Brave AI grounding failed for '%s': %s", query[:60], exc)
        return None


# ---------------------------------------------------------------------------
# DuckDuckGo fallback
# ---------------------------------------------------------------------------

def _search_ddg_news(
    topic: str,
    keywords: list[str],
    max_results: int = 5,
    time_range: str = "d",
) -> list[dict[str, Any]]:
    """DuckDuckGo fallback when Brave is unavailable."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.error("duckduckgo-search not installed: pip install duckduckgo-search")
        return []

    query = " ".join(keywords)
    results: list[dict[str, Any]] = []

    try:
        with DDGS() as ddgs:
            news_results = ddgs.news(query, max_results=max_results, timelimit=time_range)
            for item in news_results:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("body", ""),
                    "date": item.get("date", ""),
                    "source": item.get("source", ""),
                    "thumbnail_url": item.get("image", ""),
                    "favicon_url": "",
                    "extra_snippets": [],
                })
    except Exception as exc:
        logger.warning("DuckDuckGo news failed for '%s': %s", topic, exc)
        try:
            with DDGS() as ddgs:
                text_results = ddgs.text(
                    f"{query} news {datetime.now().strftime('%Y')}",
                    max_results=max_results,
                    timelimit=time_range,
                )
                for item in text_results:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                        "date": "",
                        "source": "",
                        "thumbnail_url": "",
                        "favicon_url": "",
                        "extra_snippets": [],
                    })
        except Exception as exc2:
            logger.error("DuckDuckGo text search also failed for '%s': %s", topic, exc2)

    logger.info("DDG search '%s': %d results", topic, len(results))
    return results


# ---------------------------------------------------------------------------
# Unified public API
# ---------------------------------------------------------------------------

def search_news(
    topic: str,
    keywords: list[str],
    max_results: int = 5,
    time_range: str = "d",
) -> list[dict[str, Any]]:
    """Search for recent news on a topic.

    Tries Brave Search API first (richer metadata, thumbnails),
    falls back to DuckDuckGo if Brave key is missing or API fails.

    Returns list of dicts with keys:
        title, url, snippet, date, source, thumbnail_url, favicon_url, extra_snippets
    """
    freshness = _FRESHNESS_MAP.get(time_range, "pd")

    results = _search_brave_news(topic, keywords, max_results, freshness)

    if not results:
        results = _search_brave_web(topic, keywords, max_results, freshness)

    if not results:
        results = _search_ddg_news(topic, keywords, max_results, time_range)

    return results


def search_all_topics(
    topics: list[dict[str, Any]],
    use_ai_grounding: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Search news for all configured topics.

    Parameters
    ----------
    topics:
        List of topic dicts from topics.yaml, each with name, keywords, depth.
    use_ai_grounding:
        If True and Brave AI key is available, add an AI-grounded summary
        to each topic's results under the ``ai_summary`` key.

    Returns
    -------
    Dict mapping topic name to list of search result dicts.
    """
    all_results: dict[str, list[dict[str, Any]]] = {}
    seen_urls: set[str] = set()

    for topic in topics:
        name = topic.get("name", "Unknown")
        keywords = topic.get("keywords", [])
        depth = topic.get("depth", "brief")
        max_results = 5 if depth == "detailed" else 3

        if not keywords:
            logger.warning("Topic '%s' has no keywords, skipping", name)
            continue

        results = search_news(name, keywords, max_results=max_results)

        deduped: list[dict[str, Any]] = []
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(r)

        if use_ai_grounding and deduped:
            summary = summarize_with_brave_ai(
                f"Latest news about {name}: {' '.join(keywords)}"
            )
            if summary:
                for r in deduped:
                    r["ai_summary"] = summary

        all_results[name] = deduped

    return all_results
