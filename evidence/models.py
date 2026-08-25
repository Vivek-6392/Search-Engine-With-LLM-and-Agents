from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    claim: str
    supporting_text: str = ""
    source_url: str = ""
    source_title: str = ""
    source_type: str = "web"  # web, academic, finance, code, utility, general
    confidence: float = 0.85
    node_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "supporting_text": self.supporting_text[:250] if self.supporting_text else "",
            "source_url": self.source_url,
            "source_title": self.source_title or "Source",
            "source_type": self.source_type,
            "confidence": round(self.confidence, 2),
            "node_id": self.node_id,
            "timestamp": self.timestamp,
        }


@dataclass
class ResearchPacket:
    question: str
    key_findings: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)
    uncertainty: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "key_findings": self.key_findings,
            "conflicts": self.conflicts,
            "citations": self.citations,
            "uncertainty": self.uncertainty,
            "missing_information": self.missing_information,
            "estimated_tokens": self.estimated_tokens(),
        }

    def estimated_tokens(self) -> int:
        """Estimate token count of the research packet payload."""
        md = self.to_markdown()
        return max(1, len(md) // 4)

    def to_markdown(self) -> str:
        sections = [f"## Research Target\n{self.question.strip()}"]

        if self.key_findings:
            sections.append("### Key Verified Findings\n" + "\n".join([f"- {f.strip()}" for f in self.key_findings]))

        if self.conflicts:
            sections.append("### Contradictions / Conflicting Data\n" + "\n".join([f"- ⚠️ {c.strip()}" for c in self.conflicts]))

        if self.uncertainty:
            sections.append("### Data Uncertainties\n" + "\n".join([f"- ❓ {u.strip()}" for u in self.uncertainty]))

        if self.missing_information:
            sections.append("### Missing / Unanswered Aspects\n" + "\n".join([f"- 🔍 {m.strip()}" for m in self.missing_information]))

        if self.citations:
            cit_lines = []
            for c in self.citations:
                title = c.get("title") or "Source"
                url = c.get("url") or ""
                if url:
                    cit_lines.append(f"- [{title}]({url})")
                else:
                    cit_lines.append(f"- {title}")
            sections.append("### Primary Citations & References\n" + "\n".join(cit_lines))

        return "\n\n".join(sections)
