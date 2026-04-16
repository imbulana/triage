import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ALLOWED_BRIDGE_TYPES = {
    "maps_to_issue",
    "cites_policy",
    "handled_by_team",
    "requires_sla",
    "similar_pattern",
    "depends_on",
}


@dataclass
class BridgeEdge:
    from_kg: str
    from_node: str
    to_kg: str
    to_node: str
    bridge_type: str
    confidence: float
    provenance: Dict[str, Any]
    last_validated_at: str

    def validate(self) -> None:
        if self.bridge_type not in ALLOWED_BRIDGE_TYPES:
            raise ValueError(f"Unsupported bridge_type: {self.bridge_type}")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.provenance, dict) or "method" not in self.provenance:
            raise ValueError("provenance must contain method")


class BridgeStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load(self) -> List[BridgeEdge]:
        rows = json.loads(self.path.read_text(encoding="utf-8"))
        edges: List[BridgeEdge] = []
        for row in rows:
            edge = BridgeEdge(**row)
            edge.validate()
            edges.append(edge)
        return edges

    def save(self, edges: Iterable[BridgeEdge]) -> None:
        data = []
        for edge in edges:
            edge.validate()
            data.append(asdict(edge))
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_many(self, edges: Iterable[BridgeEdge]) -> None:
        existing = self.load()
        seen = {
            (e.from_kg, e.from_node, e.to_kg, e.to_node, e.bridge_type)
            for e in existing
        }
        for edge in edges:
            edge.validate()
            key = (edge.from_kg, edge.from_node, edge.to_kg, edge.to_node, edge.bridge_type)
            if key not in seen:
                existing.append(edge)
                seen.add(key)
        self.save(existing)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
