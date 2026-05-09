import json
import logging
import re
from hashlib import md5
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

DOMAIN_TERMS = {
    "kg_complaints_core": [
        "complaint",
        "consumer",
        "product",
        "issue",
        "sub-issue",
        "narrative",
        "company response",
    ],
    "kg_credit_domain": [
        "credit card",
        "credit report",
        "billing error",
        "truth in lending",
        "interest rate",
        "unauthorized charge",
        "identity theft",
    ],
    "kg_lending_domain": [
        "loan",
        "mortgage",
        "student loan",
        "servicing",
        "escrow",
        "foreclosure",
        "payment application",
    ],
    "kg_banking_domain": [
        "bank account",
        "deposit account",
        "transfer",
        "ACH",
        "electronic fund transfer",
        "debit card",
        "overdraft",
    ],
    "kg_regulatory_policy": [
        "CFPB",
        "regulation",
        "compliance",
        "UDAAP",
        "fair lending",
        "error resolution",
        "response timeliness",
        "escalation",
    ],
}

GENERIC_TERMS = [
    "fraud",
    "discrimination",
    "harassment",
    "unauthorized",
    "investigation",
    "remediation",
    "customer response",
    "owner team",
]


def _read_chunks(chunks_file: str) -> List[dict]:
    path = Path(chunks_file)
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows if isinstance(rows, list) else []


def _normalize_entity(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" .,:;()[]{}\"'")).strip()


def _candidate_entities(text: str, kg_id: str) -> List[str]:
    lowered = text.lower()
    terms = DOMAIN_TERMS.get(kg_id, []) + GENERIC_TERMS
    candidates = []
    for term in terms:
        if term.lower() in lowered:
            candidates.append(_normalize_entity(term))

    for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9&.-]+(?:\s+|$)){2,5}", text):
        entity = _normalize_entity(match.group(0))
        if 3 <= len(entity) <= 80:
            candidates.append(entity)

    seen = set()
    deduped = []
    for entity in candidates:
        key = entity.lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(entity)
    return deduped[:12]


def _source_id(row: dict, index: int) -> str:
    text = row.get("text", "") or f"chunk-{index}"
    return str(row.get("hash_code") or md5(text.encode("utf-8")).hexdigest())


def bootstrap_kg_from_chunks(
    chunks_file: str,
    working_dir: str,
    kg_id: str,
    verbose_logging: bool = False,
) -> Dict[str, int]:
    """Create minimal entity/relation JSONL files from prepared chunks.

    This is a deterministic fallback for local development and CI. The fuller
    LeanRAG clustering/indexing path can still run afterwards when Milvus,
    MySQL, embeddings, and LLM credentials are available.
    """
    chunks = _read_chunks(chunks_file)
    output_dir = Path(working_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if verbose_logging:
        logger.info(
            "Bootstrapping deterministic KG for kg=%s from %s chunks in %s",
            kg_id,
            len(chunks),
            chunks_file,
        )

    entities: Dict[str, dict] = {}
    relations: Dict[Tuple[str, str], dict] = {}
    entity_counts = Counter()
    entity_sources = defaultdict(list)

    for index, row in enumerate(chunks):
        text = row.get("text", "") or ""
        source_id = _source_id(row, index)
        candidates = _candidate_entities(text, kg_id)
        for entity in candidates:
            entity_counts[entity] += 1
            entity_sources[entity].append(source_id)
            existing = entities.get(entity)
            sentence = text[:280].strip()
            if existing:
                if sentence and sentence not in existing["description"]:
                    existing["description"] = f"{existing['description']} | {sentence}"[:1800]
            else:
                entities[entity] = {
                    "entity_name": entity,
                    "entity_type": "concept",
                    "description": sentence or f"Concept observed in {kg_id}",
                    "source_id": source_id,
                    "degree": 0,
                }

        for left, right in _pairwise(candidates[:6]):
            key = (left, right)
            if key not in relations:
                relations[key] = {
                    "src_tgt": left,
                    "tgt_src": right,
                    "description": f"{left} appears in the same complaint knowledge context as {right}.",
                    "weight": 1,
                    "source_id": source_id,
                }
            else:
                relations[key]["weight"] += 1

    for entity, row in entities.items():
        sources = list(dict.fromkeys(entity_sources[entity]))
        row["source_id"] = "|".join(sources[:10])
        row["degree"] = entity_counts[entity]

    if verbose_logging:
        logger.info("Writing bootstrapped entity file to %s", output_dir / "entity.jsonl")
    _write_jsonl(list(entities.values()), output_dir / "entity.jsonl")
    if verbose_logging:
        logger.info("Writing bootstrapped relation file to %s", output_dir / "relation.jsonl")
    _write_jsonl(list(relations.values()), output_dir / "relation.jsonl")
    if verbose_logging:
        logger.info(
            "Bootstrap complete for kg=%s: entities=%s relations=%s",
            kg_id,
            len(entities),
            len(relations),
        )
    return {"chunks": len(chunks), "entities": len(entities), "relations": len(relations)}


def _pairwise(values: Iterable[str]) -> Iterable[Tuple[str, str]]:
    items = list(values)
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            if left != right:
                yield left, right


def _write_jsonl(rows: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
