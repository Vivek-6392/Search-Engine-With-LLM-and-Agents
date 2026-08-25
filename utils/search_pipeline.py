from dataclasses import dataclass, field
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "ref_src",
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "msclkid",
    "session_id",
    "source",
}

DISALLOWED_DOMAINS = {
    "pinterest.com",
    "pinterest.co.uk",
    "quora.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
}

TRUSTED_DOMAINS = {
    "wikipedia.org": 0.25,
    "arxiv.org": 0.30,
    "nih.gov": 0.30,
    "ncbi.nlm.nih.gov": 0.30,
    "github.com": 0.20,
    "docs.python.org": 0.25,
    "developer.mozilla.org": 0.25,
    "stackoverflow.com": 0.20,
    ".gov": 0.25,
    ".edu": 0.20,
}


@dataclass
class SearchEvidence:
    title: str
    url: str
    source: str
    snippet: str
    relevance: float
    facts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "snippet": self.snippet,
            "relevance": round(self.relevance, 3),
            "facts": self.facts,
        }

    def to_markdown(self) -> str:
        domain_tag = f" ({self.source})" if self.source else ""
        lines = [f"• **[{self.title}]({self.url})**{domain_tag} — *Relevance: {self.relevance:.0%}*"]
        if self.facts:
            for fact in self.facts[:2]:
                lines.append(f"  - {fact.strip()}")
        elif self.snippet:
            lines.append(f"  - {self.snippet.strip()}")
        return "\n".join(lines)


class RequestSearchCache:
    """Thread-safe request-scoped search cache with TTL."""

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 256):
        self._cache: Dict[str, Tuple[float, List[SearchEvidence]]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _normalize_key(self, query: str, request_id: Optional[str] = None) -> str:
        clean_q = " ".join(query.strip().lower().split())
        req = (request_id or "").strip()
        return f"{req}::{clean_q}" if req else clean_q

    def get(self, query: str, request_id: Optional[str] = None) -> Optional[List[SearchEvidence]]:
        key = self._normalize_key(query, request_id)
        with self._lock:
            if key in self._cache:
                timestamp, results = self._cache[key]
                if (time.perf_counter() - timestamp) <= self._ttl:
                    return results
                del self._cache[key]
        return None

    def set(self, query: str, results: List[SearchEvidence], request_id: Optional[str] = None):
        key = self._normalize_key(query, request_id)
        with self._lock:
            if len(self._cache) >= self._max_size:
                # Evict oldest entry
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.perf_counter(), results)

    def clear(self):
        with self._lock:
            self._cache.clear()


_GLOBAL_SEARCH_CACHE = RequestSearchCache()


def get_search_cache() -> RequestSearchCache:
    return _GLOBAL_SEARCH_CACHE


def canonicalize_url(url: str) -> str:
    """
    Normalize and canonicalize URL:
    - Validate valid http/https scheme and netloc
    - Lowercase scheme and netloc
    - Strip tracking query parameters (utm_*, ref, fbclid, etc.)
    - Remove anchor fragments (#...)
    - Strip trailing slashes and www prefix
    """
    if not url:
        return ""

    raw_url = url.strip()
    # If scheme missing entirely (e.g. example.com/path), prefix https://
    if "://" not in raw_url:
        raw_url = "https://" + raw_url

    try:
        parsed = urllib.parse.urlparse(raw_url)
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            return ""

        netloc = parsed.netloc.lower()
        if not netloc or "." not in netloc:
            return ""

        # Remove www.
        if netloc.startswith("www."):
            netloc = netloc[4:]

        # Filter tracking params
        query_params = urllib.parse.parse_qs(parsed.query, keep_blank_values=False)
        clean_params = {k: v for k, v in query_params.items() if k.lower() not in TRACKING_PARAMS}
        clean_query = urllib.parse.urlencode(clean_params, doseq=True)

        path = parsed.path.rstrip("/") if parsed.path != "/" else "/"

        canonical = urllib.parse.urlunparse(
            (scheme, netloc, path, parsed.params, clean_query, "")
        )
        return canonical
    except Exception:
        return ""


def extract_source_name(url: str) -> str:
    """Extract clean domain/source publisher name from URL."""
    try:
        netloc = urllib.parse.urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        parts = netloc.split(".")
        if len(parts) >= 2:
            return parts[-2].capitalize()
        return netloc
    except Exception:
        return ""


def deduplicate_results(raw_results: List[dict]) -> List[dict]:
    """Deduplicate search items by canonical URL and title similarity."""
    seen_urls: Set[str] = set()
    seen_titles: Set[str] = set()
    deduped: List[dict] = []

    for item in raw_results:
        raw_url = str(item.get("url") or item.get("href") or "").strip()
        if not raw_url:
            continue

        canon_url = canonicalize_url(raw_url)
        if not canon_url or canon_url in seen_urls:
            continue

        domain = urllib.parse.urlparse(canon_url).netloc.lower()
        if any(d in domain for d in DISALLOWED_DOMAINS):
            continue

        raw_title = str(item.get("title", "")).strip()
        norm_title = re.sub(r"[^a-zA-Z0-9\s]", "", raw_title.lower()).strip()
        if norm_title and norm_title in seen_titles:
            continue

        seen_urls.add(canon_url)
        if norm_title:
            seen_titles.add(norm_title)

        snippet = str(item.get("content") or item.get("snippet") or item.get("body") or "").strip()

        deduped.append(
            {
                "title": raw_title or extract_source_name(canon_url) or "Web Reference",
                "url": canon_url,
                "source": extract_source_name(canon_url),
                "snippet": snippet,
            }
        )

    return deduped


def rank_evidence_results(query: str, results: List[dict]) -> List[dict]:
    """
    Lightweight relevance ranking:
    - Title term overlap (weight 3.0)
    - Snippet term overlap (weight 1.5)
    - Domain credibility boost (+0.2 to +0.3)
    """
    if not results:
        return []

    q_terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    if not q_terms:
        q_terms = query.lower().split()

    scored_results = []
    for item in results:
        title_lower = item.get("title", "").lower()
        snippet_lower = item.get("snippet", "").lower()
        url_lower = item.get("url", "").lower()

        # Score components
        title_matches = sum(1 for t in q_terms if t in title_lower)
        snippet_matches = sum(1 for t in q_terms if t in snippet_lower)

        raw_score = (title_matches * 3.0) + (snippet_matches * 1.5)

        # Domain boost
        domain_boost = 0.0
        for trusted, boost in TRUSTED_DOMAINS.items():
            if trusted in url_lower:
                domain_boost += boost
                break

        # Max possible query score
        max_possible = max(1.0, len(q_terms) * 4.5)
        relevance_norm = min(1.0, max(0.1, (raw_score / max_possible) + domain_boost))

        scored_item = dict(item)
        scored_item["relevance"] = relevance_norm
        scored_results.append(scored_item)

    # Sort descending by relevance
    scored_results.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
    return scored_results


def extract_compact_facts(snippet: str, query: str, max_facts: int = 2) -> List[str]:
    """Extract dense, key factual sentences matching query terms."""
    if not snippet:
        return []

    # Split into clean sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", snippet) if len(s.strip()) > 20]
    if not sentences:
        return [snippet[:300].strip()]

    q_terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    if not q_terms:
        return [sentences[0][:300]]

    # Rank sentences by term density
    scored_sentences = []
    for s in sentences:
        s_lower = s.lower()
        match_count = sum(1 for t in q_terms if t in s_lower)
        scored_sentences.append((match_count, s))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    facts = []
    for count, s in scored_sentences[:max_facts]:
        clean_fact = s if len(s) <= 300 else s[:297] + "..."
        facts.append(clean_fact)

    return facts if facts else [snippet[:300]]


def _fetch_raw_search(query: str, max_raw: int = 5) -> List[dict]:
    """Fetch raw search items from Tavily or DuckDuckGo."""
    clean_q = query.strip()
    raw_items = []

    # 1. Tavily if key present
    api_key = os.getenv("TAVILY_API_KEY", "")
    if api_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            resp = client.search(query=clean_q, max_results=max_raw)
            for r in resp.get("results", []):
                raw_items.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                    }
                )
            if raw_items:
                return raw_items
        except Exception:
            pass

    # 2. DuckDuckGo fallback
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(clean_q, max_results=max_raw))
            for r in results:
                raw_items.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "content": r.get("body", ""),
                    }
                )
    except Exception:
        pass

    return raw_items


def search_and_extract_evidence(
    query: str,
    max_results: int = 3,
    max_fetch: int = 2,
    request_id: Optional[str] = None,
    cache: Optional[RequestSearchCache] = None,
) -> List[SearchEvidence]:
    """
    Complete Optimized Web Search & Evidence Pipeline:
    query
      → search cache check
      → fetch top 5 raw search results
      → canonicalize URLs
      → deduplicate
      → lightweight relevance ranking
      → select top candidates (top 2-3)
      → extract compact factual sentences
      → return normalized SearchEvidence objects
    """
    clean_q = query.strip()
    if not clean_q:
        return []

    c = cache or _GLOBAL_SEARCH_CACHE
    cached_evidence = c.get(clean_q, request_id=request_id)
    if cached_evidence is not None:
        return cached_evidence

    # 1. Fetch raw search results
    raw_results = _fetch_raw_search(clean_q, max_raw=5)
    if not raw_results:
        return []

    # 2. Deduplicate and clean URLs
    deduped = deduplicate_results(raw_results)

    # 3. Lightweight Relevance Ranking
    ranked = rank_evidence_results(clean_q, deduped)

    # 4. Limit to top candidates (max 2-3)
    top_candidates = ranked[:max_results]

    evidence_list: List[SearchEvidence] = []
    for item in top_candidates:
        facts = extract_compact_facts(item.get("snippet", ""), query=clean_q, max_facts=2)
        ev = SearchEvidence(
            title=item.get("title", ""),
            url=item.get("url", ""),
            source=item.get("source", ""),
            snippet=item.get("snippet", ""),
            relevance=float(item.get("relevance", 0.5)),
            facts=facts,
        )
        evidence_list.append(ev)

    # 5. Cache result
    c.set(clean_q, evidence_list, request_id=request_id)
    return evidence_list
