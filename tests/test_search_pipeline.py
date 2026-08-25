import unittest.mock as mock
import pytest

from utils.search_pipeline import (
    SearchEvidence,
    RequestSearchCache,
    canonicalize_url,
    deduplicate_results,
    rank_evidence_results,
    extract_compact_facts,
    search_and_extract_evidence,
)


def test_url_normalization_and_canonicalization():
    # Tracking parameters removal
    url_with_utm = "https://www.example.com/article/?utm_source=twitter&utm_medium=social&utm_campaign=launch&ref=techcrunch#section-2"
    canon = canonicalize_url(url_with_utm)
    assert "utm_source" not in canon
    assert "utm_medium" not in canon
    assert "ref" not in canon
    assert "#section-2" not in canon
    assert "www." not in canon
    assert canon == "https://example.com/article"

    # Trailing slash normalization
    assert canonicalize_url("https://example.com/docs/") == "https://example.com/docs"
    # Root path preservation
    assert canonicalize_url("https://example.com/") == "https://example.com/"


def test_deduplicate_results():
    raw_results = [
        {"title": "Apple Q3 Results", "url": "https://www.apple.com/news/q3?utm_source=feed", "content": "Record revenue"},
        {"title": "Apple Q3 Results", "url": "https://apple.com/news/q3", "content": "Record revenue duplicate"},
        {"title": "Pinterest Post", "url": "https://pinterest.com/pin/12345", "content": "Spam pin"},
        {"title": "Microsoft Cloud Growth", "url": "https://microsoft.com/news/cloud", "content": "Azure revenue jumps"},
    ]

    deduped = deduplicate_results(raw_results)
    assert len(deduped) == 2
    urls = [item["url"] for item in deduped]
    assert "https://apple.com/news/q3" in urls
    assert "https://microsoft.com/news/cloud" in urls
    assert not any("pinterest.com" in u for u in urls)


def test_relevance_ranking():
    query = "Nvidia Blackwell GPU architecture specifications"
    results = [
        {"title": "Random Recipe for Cookies", "snippet": "Bake at 350 degrees", "url": "https://recipes.com"},
        {"title": "Nvidia Blackwell GPU Architecture Deep Dive", "snippet": "Detailed specifications of the Blackwell B200 architecture", "url": "https://anandtech.com/blackwell"},
        {"title": "Nvidia Official Overview", "snippet": "Blackwell architecture delivers 30x performance", "url": "https://developer.nvidia.com"},
    ]

    ranked = rank_evidence_results(query, results)
    assert len(ranked) == 3
    # Top result must be the highly relevant Blackwell GPU architecture article
    assert "Blackwell" in ranked[0]["title"]
    assert ranked[0]["relevance"] > ranked[2]["relevance"]
    assert ranked[2]["title"] == "Random Recipe for Cookies"


def test_request_scoped_search_caching():
    cache = RequestSearchCache(ttl_seconds=60.0)

    mock_evidence = [
        SearchEvidence(
            title="FastAPI Docs",
            url="https://fastapi.tiangolo.com",
            source="Fastapi",
            snippet="High performance web framework",
            relevance=0.95,
            facts=["FastAPI is built on Starlette and Pydantic."],
        )
    ]

    # Set in cache
    cache.set("fastapi python framework", mock_evidence, request_id="req-123")

    # Cache hit
    cached = cache.get("fastapi python framework", request_id="req-123")
    assert cached is not None
    assert len(cached) == 1
    assert cached[0].title == "FastAPI Docs"

    # Case insensitivity and whitespace normalization
    cached_norm = cache.get("  FastAPI   PYTHON  Framework  ", request_id="req-123")
    assert cached_norm is not None
    assert len(cached_norm) == 1


def test_malformed_search_results_handling():
    raw_malformed = [
        {"title": None, "url": "", "content": None},
        {"title": "", "url": "invalid://not-http", "content": "something"},
        {"title": "Valid Source", "url": "https://example.org/article", "snippet": "Valid snippet text"},
    ]

    deduped = deduplicate_results(raw_malformed)
    assert len(deduped) == 1
    assert deduped[0]["title"] == "Valid Source"


def test_result_limits_and_pipeline_execution():
    with mock.patch("utils.search_pipeline._fetch_raw_search") as mock_fetch:
        mock_fetch.return_value = [
            {"title": f"Result {i}", "url": f"https://example.com/page{i}", "content": f"Fact snippet {i} for quantum computing"}
            for i in range(1, 10)
        ]

        cache = RequestSearchCache()
        # Request top 2 results
        evidence = search_and_extract_evidence(
            query="quantum computing breakthroughs",
            max_results=2,
            cache=cache,
        )

        assert len(evidence) == 2
        for ev in evidence:
            assert isinstance(ev, SearchEvidence)
            assert ev.url.startswith("https://example.com")
            assert len(ev.facts) >= 1


def test_normalized_evidence_schema():
    ev = SearchEvidence(
        title="ArXiv Preprint on Transformers",
        url="https://arxiv.org/abs/2301.00000",
        source="Arxiv",
        snippet="Transformers achieve state-of-the-art results on benchmark tasks.",
        relevance=0.92,
        facts=["Transformers achieve state-of-the-art results on benchmark tasks."],
    )

    ev_dict = ev.to_dict()
    assert ev_dict["title"] == "ArXiv Preprint on Transformers"
    assert ev_dict["url"] == "https://arxiv.org/abs/2301.00000"
    assert ev_dict["source"] == "Arxiv"
    assert ev_dict["relevance"] == 0.92
    assert len(ev_dict["facts"]) == 1

    md = ev.to_markdown()
    assert "[ArXiv Preprint on Transformers](https://arxiv.org/abs/2301.00000)" in md
    assert "Relevance: 92%" in md
