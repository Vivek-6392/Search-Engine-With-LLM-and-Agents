import pytest

from evidence.models import EvidenceItem, ResearchPacket
from evidence.store import EvidenceStore, build_research_packet


def test_evidence_item_creation_and_dict():
    item = EvidenceItem(
        claim="Nvidia's Blackwell B200 GPU contains 208 billion transistors.",
        supporting_text="The Blackwell B200 features 208 billion transistors manufactured on TSMC 4NP process.",
        source_url="https://nvidianews.nvidia.com/blackwell",
        source_title="Nvidia Newsroom",
        source_type="hardware",
        confidence=0.95,
        node_id="node_1",
    )

    d = item.to_dict()
    assert d["claim"] == "Nvidia's Blackwell B200 GPU contains 208 billion transistors."
    assert d["source_url"] == "https://nvidianews.nvidia.com/blackwell"
    assert d["source_title"] == "Nvidia Newsroom"
    assert d["confidence"] == 0.95
    assert d["node_id"] == "node_1"


def test_evidence_store_add_and_html_cleaning():
    store = EvidenceStore()

    raw_html_output = """
    <div>
        <h2>Quarterly Financial Highlights</h2>
        <p>Apple Inc reported quarterly revenue of <b>$94.9 billion</b>, up 6 percent year over year.</p>
        <span>Services revenue reached an all-time record of $25.0 billion.</span>
    </div>
    """

    store.add_from_raw(
        raw_html_output,
        node_id="node_apple",
        source_url="https://apple.com/investor",
        source_title="Apple IR",
    )

    assert len(store) >= 2
    claims = [it.claim for it in store.top_evidence()]
    # HTML tags should be cleanly removed
    assert not any("<" in c or ">" in c for c in claims)
    assert any("94.9 billion" in c for c in claims)
    assert any("25.0 billion" in c for c in claims)


def test_evidence_store_deduplication():
    store = EvidenceStore()

    # Exact and near-duplicate claims
    store.add(EvidenceItem(claim="Apple market cap reached $3.5 trillion in 2024."))
    store.add(EvidenceItem(claim="Apple market cap reached $3.5 trillion in 2024."))
    store.add(EvidenceItem(claim="apple market cap reached 3.5 trillion in 2024"))
    store.add(EvidenceItem(claim="Microsoft reported strong Azure cloud revenue growth."))

    assert len(store) == 4
    store.deduplicate()
    assert len(store) == 2


def test_evidence_store_merge_and_filter():
    store_a = EvidenceStore()
    store_a.add(EvidenceItem(claim="Tesla Model Y was the best-selling vehicle globally.", confidence=0.90, source_type="auto"))
    store_a.add(EvidenceItem(claim="Rumored unverified Tesla phone release in 2025.", confidence=0.40, source_type="rumor"))

    store_b = EvidenceStore()
    store_b.add(EvidenceItem(claim="BYD surpassed Tesla in battery EV production in Q4.", confidence=0.88, source_type="auto"))

    store_a.merge(store_b)
    assert len(store_a) == 3

    # Filter out low confidence
    high_conf = store_a.filter(min_confidence=0.80)
    assert len(high_conf) == 2
    assert not any("phone" in it.claim.lower() for it in high_conf)


def test_large_raw_output_compacted_to_bounded_tokens():
    store = EvidenceStore()

    # Simulate 50KB / ~15,000 tokens of verbose web dumps across 4 nodes
    for node_idx in range(1, 5):
        verbose_dump = []
        for i in range(100):
            verbose_dump.append(
                f"<div class='ad-block'>Result {i}: Extensive verbose paragraph explaining repetitive details "
                f"about tech company {node_idx} metric {i}. Source: [TechNews](https://technews.com/article/{node_idx}_{i}).</div>"
            )
        raw_text = "\n".join(verbose_dump)
        store.add_from_raw(raw_text, node_id=f"node_{node_idx}")

    # Initial raw text is enormous
    assert len(store) >= 100

    # Build ResearchPacket
    packet = build_research_packet(
        question="Compare major tech companies Q3 performance and cloud revenue",
        store=store,
        max_sources=6,
        max_findings=12,
    )

    # Token count of the research packet must be strictly <= 5,000 tokens (target is well below 1,500)
    est_tokens = packet.estimated_tokens()
    assert est_tokens < 5000
    assert len(packet.key_findings) <= 12
    assert len(packet.citations) <= 6


def test_build_research_packet_schema_and_markdown():
    store = EvidenceStore()
    store.add(
        EvidenceItem(
            claim="Global EV sales grew 25% year-over-year in 2024.",
            source_url="https://iea.org/reports/global-ev-outlook",
            source_title="IEA Report",
            confidence=0.95,
        )
    )
    store.add(
        EvidenceItem(
            claim="Battery pack costs fell below $100 per kWh.",
            source_url="https://bloomberg.com/bnef",
            source_title="BloombergNEF",
            confidence=0.90,
        )
    )

    packet = build_research_packet(
        question="What are the latest EV market trends and battery cost benchmarks?",
        store=store,
    )

    assert packet.question == "What are the latest EV market trends and battery cost benchmarks?"
    assert len(packet.key_findings) == 2
    assert len(packet.citations) == 2

    md = packet.to_markdown()
    assert "## Research Target" in md
    assert "### Key Verified Findings" in md
    assert "Global EV sales grew 25%" in md
    assert "[IEA Report](https://iea.org/reports/global-ev-outlook)" in md
