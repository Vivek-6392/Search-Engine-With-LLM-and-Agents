from collections import defaultdict
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from evidence.models import EvidenceItem, ResearchPacket


def _clean_text(raw: str) -> str:
    """Strip raw HTML, JSON brackets, and excessive formatting."""
    if not raw:
        return ""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", str(raw))
    # Strip markdown header prefixes and bullet markers
    clean = re.sub(r"^[#\*\-•\s]+", "", clean, flags=re.MULTILINE)
    # Strip repeated whitespace
    return " ".join(clean.split()).strip()


def _normalize_claim(claim: str) -> str:
    """Normalize claim for duplicate detection."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", claim.lower())
    return " ".join(clean.split())


def _extract_url_from_text(text: str) -> Optional[Tuple[str, str]]:
    """Extract first markdown link [Title](url) from text if present."""
    match = re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None


class EvidenceStore:
    """
    Thread-safe storage and compaction engine for structured research evidence.
    """

    def __init__(self):
        self._items: List[EvidenceItem] = []
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def add(self, item: EvidenceItem):
        """Add a sanitized EvidenceItem to the store."""
        with self._lock:
            clean_claim = _clean_text(item.claim)
            if not clean_claim or len(clean_claim) < 4:
                return

            clean_supp = _clean_text(item.supporting_text)[:250] if item.supporting_text else ""
            item.claim = clean_claim
            item.supporting_text = clean_supp
            self._items.append(item)

    def add_from_raw(
        self,
        raw_text: str,
        node_id: str = "",
        source_type: str = "web",
        source_url: str = "",
        source_title: str = "",
        confidence: float = 0.85,
    ):
        """
        Parse raw node output/search text, extract discrete factual claims,
        and ingest them into the EvidenceStore.
        """
        if not raw_text or not raw_text.strip():
            return

        with self._lock:
            lines = raw_text.splitlines()
            for line in lines:
                line_clean = line.strip()
                if not line_clean or len(line_clean) < 4:
                    continue

                # Ignore tool headers and boilerplate lines
                if any(
                    line_clean.lower().startswith(bp)
                    for bp in [
                        "result from",
                        "search provider:",
                        "tavily error:",
                        "ddgs error:",
                        "web search error:",
                        "unable to retrieve",
                    ]
                ):
                    continue

                url = source_url
                title = source_title
                link_match = _extract_url_from_text(line_clean)
                if link_match:
                    title = link_match[0]
                    url = link_match[1]

                cleaned_claim = _clean_text(line_clean)
                if not cleaned_claim:
                    continue

                fallback_title = title or (f"Node {node_id}" if node_id else "Reference")

                # If line is structured key-value data or short fact, preserve as whole claim
                if len(cleaned_claim) < 140 or ":" in cleaned_claim[:30]:
                    self.add(
                        EvidenceItem(
                            claim=cleaned_claim,
                            supporting_text=cleaned_claim[:200],
                            source_url=url,
                            source_title=fallback_title,
                            source_type=source_type,
                            confidence=confidence,
                            node_id=node_id,
                        )
                    )
                else:
                    # Long narrative text: split on sentence boundaries
                    sentences = [
                        s.strip() for s in re.split(r"(?<=[a-zA-Z0-9][.!?])\s+(?=[A-Z])", cleaned_claim) if len(s.strip()) >= 6
                    ]
                    for sent in sentences:
                        self.add(
                            EvidenceItem(
                                claim=sent,
                                supporting_text=cleaned_claim[:200],
                                source_url=url,
                                source_title=fallback_title,
                                source_type=source_type,
                                confidence=confidence,
                                node_id=node_id,
                            )
                        )

    def deduplicate(self):
        """Deduplicate claims and overlapping snippets."""
        with self._lock:
            seen_claims: Set[str] = set()
            unique_items: List[EvidenceItem] = []

            for item in self._items:
                norm = _normalize_claim(item.claim)
                # Check exact or prefix overlap
                if not norm or norm in seen_claims:
                    continue

                # Fuzzy token overlap check
                norm_tokens = set(norm.split())
                is_duplicate = False
                for seen in list(seen_claims):
                    seen_tokens = set(seen.split())
                    intersection = norm_tokens.intersection(seen_tokens)
                    smaller_len = min(len(norm_tokens), len(seen_tokens))
                    if smaller_len > 3 and (len(intersection) / smaller_len) >= 0.85:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    seen_claims.add(norm)
                    unique_items.append(item)

            self._items = unique_items

    def merge(self, other: "EvidenceStore"):
        """Merge items from another EvidenceStore."""
        with self._lock:
            with other._lock:
                for item in other._items:
                    self._items.append(item)
            self.deduplicate()

    def filter(
        self,
        min_confidence: float = 0.0,
        source_types: Optional[List[str]] = None,
        node_ids: Optional[List[str]] = None,
    ) -> List[EvidenceItem]:
        """Filter evidence items by confidence, source type, and node ID."""
        with self._lock:
            filtered = []
            for item in self._items:
                if item.confidence < min_confidence:
                    continue
                if source_types and item.source_type not in source_types:
                    continue
                if node_ids and item.node_id not in node_ids:
                    continue
                filtered.append(item)
            return filtered

    def top_evidence(self, max_items: int = 20) -> List[EvidenceItem]:
        """Return highest-confidence, diverse evidence items across nodes and sources."""
        with self._lock:
            self.deduplicate()
            # Sort by confidence descending
            sorted_items = sorted(self._items, key=lambda x: x.confidence, reverse=True)
            # Ensure diversity across distinct nodes / URLs
            selected = []
            seen_keys = defaultdict(int)
            for item in sorted_items:
                key = (item.node_id, item.source_url or item.source_title)
                if seen_keys[key] < 8:
                    selected.append(item)
                    seen_keys[key] += 1
                if len(selected) >= max_items:
                    break
            return selected

    def compact(self, max_total_chars: int = 6000) -> List[EvidenceItem]:
        """
        Compact evidence items to strictly fit within token bounds (<= 1.5K tokens / 6K chars).
        """
        with self._lock:
            top_items = self.top_evidence(max_items=24)
            compacted = []
            current_chars = 0

            for item in top_items:
                item_len = len(item.claim) + len(item.supporting_text) + len(item.source_title) + 50
                if current_chars + item_len <= max_total_chars:
                    compacted.append(item)
                    current_chars += item_len
                else:
                    break

            return compacted

    def to_markdown(self, max_items: int = 12) -> str:
        """Format top evidence items into concise markdown list."""
        with self._lock:
            items = self.compact(max_total_chars=5000)[:max_items]
            if not items:
                return "No structured evidence recorded."

            lines = []
            for it in items:
                source_tag = f" — [{it.source_title}]({it.source_url})" if it.source_url else f" — *{it.source_title}*"
                lines.append(f"• {it.claim}{source_tag}")
            return "\n".join(lines)


def build_research_packet(
    question: str,
    store: EvidenceStore,
    max_sources: int = 6,
    max_findings: int = 20,
) -> ResearchPacket:
    """
    Compile a structured, token-bounded ResearchPacket directly from an EvidenceStore.
    Guarantees synthesis payload is <= 5K tokens.
    """
    evidence_items = store.compact(max_total_chars=6000)

    findings = []
    citations_map: Dict[str, str] = {}  # url -> title
    conflicts = []
    uncertainty = []
    missing_info = []

    seen_claims = set()
    for item in evidence_items:
        norm = _normalize_claim(item.claim)
        if norm not in seen_claims:
            seen_claims.add(norm)
            findings.append(item.claim)

        if item.source_url:
            title = item.source_title or "Reference"
            citations_map[item.source_url] = title

        if item.confidence < 0.65:
            uncertainty.append(f"Low confidence fact ({item.confidence:.0%}): {item.claim}")

    # Detect conflicts in numerical values or direct negations if any
    claim_texts = " ".join(findings).lower()
    if "increase" in claim_texts and "decrease" in claim_texts:
        conflicts.append("Source discrepancies observed regarding metrics trajectory (increase vs decrease).")

    if not findings:
        missing_info.append("No definitive primary factual claims gathered.")

    citations = [
        {"title": title, "url": url}
        for url, title in list(citations_map.items())[:max_sources]
    ]

    return ResearchPacket(
        question=question,
        key_findings=findings[:max_findings],
        conflicts=conflicts,
        citations=citations,
        uncertainty=uncertainty,
        missing_information=missing_info,
    )
