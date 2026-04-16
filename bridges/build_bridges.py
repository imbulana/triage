import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from bridges.bridge_store import BridgeEdge, BridgeStore, now_iso


def _read_entities(entity_file: Path) -> List[dict]:
    if not entity_file.exists():
        return []
    rows = []
    with entity_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _index_entities(kg_to_entity_file: Dict[str, Path]) -> Dict[str, List[dict]]:
    return {kg: _read_entities(path) for kg, path in kg_to_entity_file.items()}


def build_candidate_bridges(kg_to_entity_file: Dict[str, Path], min_overlap_chars: int = 6) -> List[BridgeEdge]:
    all_entities = _index_entities(kg_to_entity_file)
    name_index = defaultdict(list)
    for kg_id, rows in all_entities.items():
        for row in rows:
            name = str(row.get("entity_name", "")).strip()
            if name:
                name_index[name.lower()].append((kg_id, name))

    edges: List[BridgeEdge] = []
    stamp = now_iso()
    for normalized_name, mentions in name_index.items():
        if len(mentions) < 2 or len(normalized_name) < min_overlap_chars:
            continue
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                left, right = mentions[i], mentions[j]
                if left[0] == right[0]:
                    continue
                edges.append(
                    BridgeEdge(
                        from_kg=left[0],
                        from_node=left[1],
                        to_kg=right[0],
                        to_node=right[1],
                        bridge_type="similar_pattern",
                        confidence=0.6,
                        provenance={"method": "entity_name_overlap", "sources": []},
                        last_validated_at=stamp,
                    )
                )
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-KG bridge edges")
    parser.add_argument("--kg-map", required=True, help="JSON path: {kg_id: entity_jsonl_path}")
    parser.add_argument("--output", required=True, help="Output JSON file for bridge edges")
    args = parser.parse_args()

    kg_map = json.loads(Path(args.kg_map).read_text(encoding="utf-8"))
    kg_to_entity_file = {kg: Path(path) for kg, path in kg_map.items()}
    edges = build_candidate_bridges(kg_to_entity_file)
    store = BridgeStore(args.output)
    store.add_many(edges)
    print(f"Saved {len(store.load())} bridge edges to {args.output}")


if __name__ == "__main__":
    main()
