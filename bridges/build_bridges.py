import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI

from bridges.bridge_store import BridgeEdge, BridgeStore, now_iso
from llm_settings import load_llm_settings

logger = logging.getLogger(__name__)


@dataclass
class BridgeBuildResult:
    accepted: List[BridgeEdge]
    review: List[BridgeEdge]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

GENERIC_ENTITY_NAMES = {
    "account",
    "accounts",
    "bureau",
    "consumer",
    "consumers",
    "credit",
    "data",
    "debt",
    "director",
    "financial",
    "information",
    "loan",
    "loans",
    "payment",
    "payments",
    "product",
    "products",
    "regulation",
    "service",
    "services",
}

MEANINGFUL_ACRONYMS = {
    "ach",
    "cfpa",
    "cfpb",
    "cfr",
    "ecoa",
    "fcra",
    "fdcpa",
    "hmda",
    "respa",
    "tila",
    "udaap",
    "usc",
}

KNOWN_ALIAS_CANONICALS = {
    "annual percentage rate": "apr",
    "apr": "apr",
    "bureau of consumer financial protection": "cfpb",
    "cfpb": "cfpb",
    "consumer financial protection bureau": "cfpb",
    "consumer financial protection bureau cfpb": "cfpb",
    "federal trade commission": "ftc",
    "federal trade commission ftc": "ftc",
    "ftc": "ftc",
}

ENTITY_TYPE_TERMS = {
    "act",
    "agency",
    "bureau",
    "commission",
    "issuer",
    "law",
    "rate",
    "regulation",
}

CITATION_RE = re.compile(
    r"\b(?:\d+\s*(?:cfr|u\.s\.c\.|usc)\s*[§ ]*\s*\d+(?:\.\d+)?(?:\([a-z0-9]+\))*)\b"
    r"|\bregulation\s+[a-z]{1,2}\b",
    re.IGNORECASE,
)

REGULATION_NAME_RE = re.compile(r"^regulation\s+([a-z]{1,2})$", re.IGNORECASE)
PAREN_ALIAS_RE = re.compile(r"\(([A-Za-z0-9&.-]{2,12})\)")

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


def _read_relations(relation_file: Path) -> List[dict]:
    if not relation_file.exists():
        return []
    rows = []
    with relation_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_entities(kg_to_entity_file: Dict[str, Path]) -> Dict[str, List[dict]]:
    return {kg: _read_entities(path) for kg, path in kg_to_entity_file.items()}


def _index_neighbors(kg_to_entity_file: Dict[str, Path]) -> Dict[Tuple[str, str], set[str]]:
    neighbors: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for kg_id, entity_file in kg_to_entity_file.items():
        relation_file = entity_file.parent / "relation.jsonl"
        for row in _read_relations(relation_file):
            left = str(row.get("src_tgt", "")).strip()
            right = str(row.get("tgt_src", "")).strip()
            description = str(row.get("description", ""))
            if not left or not right:
                continue
            neighbors[(kg_id, _normalize_text(left))].update(_tokens(right))
            neighbors[(kg_id, _normalize_text(right))].update(_tokens(left))
            relation_tokens = _tokens(description)
            neighbors[(kg_id, _normalize_text(left))].update(relation_tokens)
            neighbors[(kg_id, _normalize_text(right))].update(relation_tokens)
    return neighbors


def _normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = value.strip("\"'` ")
    value = re.sub(r"[^a-z0-9§.()]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_text(value))
        if len(token) > 2 and token not in STOPWORDS
    }


def _regulation_letter(value: str) -> str | None:
    match = REGULATION_NAME_RE.match(_normalize_text(value))
    return match.group(1) if match else None


def _parenthetical_aliases(value: str) -> set[str]:
    return {_normalize_text(match) for match in PAREN_ALIAS_RE.findall(value or "")}


def _canonical_alias(value: str) -> str | None:
    normalized = _normalize_text(value).replace("(", " ").replace(")", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in KNOWN_ALIAS_CANONICALS:
        return KNOWN_ALIAS_CANONICALS[normalized]
    for alias in _parenthetical_aliases(value):
        if alias in KNOWN_ALIAS_CANONICALS:
            return KNOWN_ALIAS_CANONICALS[alias]
    return None


def _citations(*values: str) -> set[str]:
    citations = set()
    for value in values:
        for match in CITATION_RE.findall(value or ""):
            citations.add(_normalize_text(match))
    return citations


def _is_bridgeable_name(mention: dict, min_overlap_chars: int) -> bool:
    normalized = mention["normalized_name"]
    if len(normalized) < min_overlap_chars:
        return False
    if mention["citations"]:
        return True
    tokens = mention["name_tokens"]
    if len(tokens) >= 2:
        return True
    if len(tokens) == 1:
        token = next(iter(tokens))
        return token in MEANINGFUL_ACRONYMS and normalized not in GENERIC_ENTITY_NAMES
    return False


def _generic_penalty(left: dict, right: dict) -> float:
    left_generic = left["normalized_name"] in GENERIC_ENTITY_NAMES or len(left["name_tokens"]) <= 1
    right_generic = right["normalized_name"] in GENERIC_ENTITY_NAMES or len(right["name_tokens"]) <= 1
    if left["citations"] or right["citations"]:
        return 0.0
    if left_generic and right_generic:
        return 0.25
    if left_generic or right_generic:
        return 0.12
    return 0.0


def _type_mismatch_penalty(left: dict, right: dict) -> float:
    left_types = left["name_tokens"] & ENTITY_TYPE_TERMS
    right_types = right["name_tokens"] & ENTITY_TYPE_TERMS
    if not left_types or not right_types or left_types == right_types:
        return 0.0
    if _canonical_alias(left["name"]) and _canonical_alias(left["name"]) == _canonical_alias(right["name"]):
        return 0.0
    return 0.18


def _is_incompatible_regulation_pair(left: dict, right: dict) -> bool:
    left_letter = _regulation_letter(left["name"])
    right_letter = _regulation_letter(right["name"])
    return bool(left_letter and right_letter and left_letter != right_letter)


def _alias_overlap(left: dict, right: dict) -> bool:
    left_aliases = _parenthetical_aliases(left["name"])
    right_aliases = _parenthetical_aliases(right["name"])
    left_normalized = left["normalized_name"]
    right_normalized = right["normalized_name"]
    left_canonical = _canonical_alias(left["name"])
    right_canonical = _canonical_alias(right["name"])
    return (
        bool(left_canonical and left_canonical == right_canonical)
        or bool(left_canonical and right_normalized == left_canonical)
        or bool(right_canonical and left_normalized == right_canonical)
        or right_normalized in left_aliases
        or left_normalized in right_aliases
        or bool(left_aliases & right_aliases)
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _source_ids(row: dict) -> List[str]:
    return [source for source in str(row.get("source_id", "")).split("|") if source]


def _edge_key(edge: BridgeEdge) -> Tuple[str, str, str, str, str]:
    return (edge.from_kg, edge.from_node, edge.to_kg, edge.to_node, edge.bridge_type)


def _canonical_edge_key(edge: BridgeEdge) -> Tuple[str, Tuple[str, str], Tuple[str, str]]:
    left = (edge.from_kg, edge.from_node)
    right = (edge.to_kg, edge.to_node)
    if right < left:
        left, right = right, left
    return (edge.bridge_type, left, right)


def _make_edge(
    *,
    left: dict,
    right: dict,
    bridge_type: str,
    confidence: float,
    method: str,
    signals: dict,
    tier: str,
    stamp: str,
) -> BridgeEdge:
    return BridgeEdge(
        from_kg=left["kg_id"],
        from_node=left["name"],
        to_kg=right["kg_id"],
        to_node=right["name"],
        bridge_type=bridge_type,
        confidence=round(min(1.0, max(0.0, confidence)), 3),
        provenance={
            "method": method,
            "tier": tier,
            "sources": sorted(set(_source_ids(left["row"]) + _source_ids(right["row"]))),
            "signals": signals,
        },
        last_validated_at=stamp,
    )


def _embedding_text(mention: dict) -> str:
    description = str(mention["row"].get("description", ""))
    return f"{mention['name']}\n{description}"[:3000]


def _choose_bridge_type(method: str, shared_citations: set[str]) -> str:
    if method == "shared_legal_citation" or shared_citations:
        return "cites_policy"
    return "similar_pattern"


def _embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    settings = load_llm_settings()
    client_kwargs = {"api_key": settings["embedding_api_key"]}
    if settings["embedding_base_url"]:
        client_kwargs["base_url"] = settings["embedding_base_url"]
    client = OpenAI(**client_kwargs)
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=settings["embedding_model"], input=batch)
        vectors.extend(item.embedding for item in response.data)
        logger.info(
            "Embedded bridge batch %s-%s/%s with model=%s",
            start + 1,
            start + len(batch),
            len(texts),
            settings["embedding_model"],
        )
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _add_semantic_edges(
    *,
    mentions: List[dict],
    edges: List[BridgeEdge],
    seen_edges: set,
    stamp: str,
    threshold: float,
    max_edges_per_entity: int,
) -> None:
    logger.info("Starting semantic bridge scan for %s entities", len(mentions))
    embeddings = _embed_texts([_embedding_text(mention) for mention in mentions])
    by_kg = defaultdict(list)
    for index, mention in enumerate(mentions):
        by_kg[mention["kg_id"]].append((index, mention))

    def add_edge(edge: BridgeEdge) -> None:
        key = _edge_key(edge)
        reverse_key = (edge.to_kg, edge.to_node, edge.from_kg, edge.from_node, edge.bridge_type)
        if key in seen_edges or reverse_key in seen_edges:
            return
        seen_edges.add(key)
        edges.append(edge)

    kg_ids = sorted(by_kg)
    for left_pos, left_kg in enumerate(kg_ids):
        for right_kg in kg_ids[left_pos + 1 :]:
            left_items = by_kg[left_kg]
            right_items = by_kg[right_kg]
            left_indices = [item[0] for item in left_items]
            right_indices = [item[0] for item in right_items]
            similarities = embeddings[left_indices] @ embeddings[right_indices].T
            for row_index, left in enumerate(left_items):
                candidate_indices = np.where(similarities[row_index] >= threshold)[0]
                if len(candidate_indices) == 0:
                    continue
                ranked = sorted(
                    candidate_indices,
                    key=lambda idx: similarities[row_index, idx],
                    reverse=True,
                )[:max_edges_per_entity]
                for col_index in ranked:
                    right = right_items[int(col_index)]
                    similarity = float(similarities[row_index, col_index])
                    add_edge(
                        _make_edge(
                            left=left[1],
                            right=right[1],
                            bridge_type="similar_pattern",
                            confidence=max(0.7, similarity),
                            method="semantic_embedding_similarity",
                            signals={"embedding_similarity": round(similarity, 3)},
                            tier="tier2_scored",
                            stamp=stamp,
                        )
                    )


def _semantic_similarity_map(mentions: List[dict]) -> Dict[Tuple[int, int], float]:
    embeddings = _embed_texts([_embedding_text(mention) for mention in mentions])
    by_kg = defaultdict(list)
    for index, mention in enumerate(mentions):
        by_kg[mention["kg_id"]].append(index)

    similarities: Dict[Tuple[int, int], float] = {}
    kg_ids = sorted(by_kg)
    for left_pos, left_kg in enumerate(kg_ids):
        for right_kg in kg_ids[left_pos + 1 :]:
            left_indices = by_kg[left_kg]
            right_indices = by_kg[right_kg]
            matrix = embeddings[left_indices] @ embeddings[right_indices].T
            for row_index, left_index in enumerate(left_indices):
                for col_index, right_index in enumerate(right_indices):
                    similarities[(min(left_index, right_index), max(left_index, right_index))] = float(
                        matrix[row_index, col_index]
                    )
    return similarities


def _score_candidate(
    left: dict,
    right: dict,
    *,
    semantic_similarity: float | None,
) -> Tuple[float, str, dict]:
    shared_citations = left["citations"] & right["citations"]
    if _is_incompatible_regulation_pair(left, right) and not shared_citations:
        return 0.0, "incompatible_regulation_letter", {
            "blocked_reason": "different_regulation_letters",
        }
    name_similarity = SequenceMatcher(None, left["normalized_name"], right["normalized_name"]).ratio()
    name_overlap = _jaccard(left["name_tokens"], right["name_tokens"])
    description_overlap = _jaccard(left["description_tokens"], right["description_tokens"])
    neighborhood_overlap = _jaccard(left["neighbor_tokens"], right["neighbor_tokens"])
    generic_penalty = _generic_penalty(left, right)
    type_penalty = _type_mismatch_penalty(left, right)
    penalty = generic_penalty + type_penalty

    alias_overlap = _alias_overlap(left, right)
    if left["normalized_name"] == right["normalized_name"]:
        score = 0.94 - penalty
        method = "normalized_name_exact"
    elif alias_overlap:
        score = 0.86 - penalty
        method = "alias_overlap"
    elif shared_citations:
        score = 0.91 - (penalty * 0.5)
        method = "shared_legal_citation"
    else:
        semantic_component = semantic_similarity or 0.0
        score = (
            0.30 * name_similarity
            + 0.20 * name_overlap
            + 0.25 * semantic_component
            + 0.15 * description_overlap
            + 0.10 * neighborhood_overlap
            - penalty
        )
        method = "weighted_evidence_alignment"

    signals = {
        "name_similarity": round(name_similarity, 3),
        "name_token_overlap": round(name_overlap, 3),
        "description_token_overlap": round(description_overlap, 3),
        "neighborhood_token_overlap": round(neighborhood_overlap, 3),
        "generic_penalty": round(generic_penalty, 3),
        "type_mismatch_penalty": round(type_penalty, 3),
    }
    if alias_overlap:
        signals["alias_overlap"] = True
    if semantic_similarity is not None:
        signals["embedding_similarity"] = round(semantic_similarity, 3)
    if shared_citations:
        signals["shared_citations"] = sorted(shared_citations)
    return max(0.0, min(1.0, score)), method, signals


def _put_best_edge(target: Dict[Tuple[str, Tuple[str, str], Tuple[str, str]], BridgeEdge], edge: BridgeEdge) -> None:
    key = _canonical_edge_key(edge)
    existing = target.get(key)
    if existing is None or edge.confidence > existing.confidence:
        target[key] = edge


def build_bridge_sets(
    kg_to_entity_file: Dict[str, Path],
    min_overlap_chars: int = 6,
    accept_threshold: float = 0.78,
    review_threshold: float = 0.58,
    semantic: bool = False,
    semantic_threshold: float = 0.86,
    max_semantic_edges_per_entity: int = 3,
    verbose_logging: bool = False,
) -> BridgeBuildResult:
    all_entities = _index_entities(kg_to_entity_file)
    neighbor_index = _index_neighbors(kg_to_entity_file)
    if verbose_logging:
        for kg_id, rows in all_entities.items():
            logger.info("Loaded %s entities for bridge scan from kg=%s", len(rows), kg_id)
    mentions = []
    for kg_id, rows in all_entities.items():
        for row in rows:
            name = str(row.get("entity_name", "")).strip()
            if name:
                description = str(row.get("description", ""))
                mentions.append(
                    {
                        "kg_id": kg_id,
                        "name": name,
                        "row": row,
                        "normalized_name": _normalize_text(name),
                        "name_tokens": _tokens(name),
                        "description_tokens": _tokens(description),
                        "citations": _citations(name, description),
                        "neighbor_tokens": neighbor_index[(kg_id, _normalize_text(name))],
                    }
                )

    stamp = now_iso()
    semantic_scores = _semantic_similarity_map(mentions) if semantic else {}
    semantic_allowed = set()
    if semantic:
        by_entity = defaultdict(list)
        for (left_index, right_index), score in semantic_scores.items():
            if score >= semantic_threshold:
                by_entity[left_index].append((score, left_index, right_index))
                by_entity[right_index].append((score, left_index, right_index))
        for candidates in by_entity.values():
            for _, left_index, right_index in sorted(candidates, reverse=True)[:max_semantic_edges_per_entity]:
                semantic_allowed.add((left_index, right_index))
    accepted: Dict[Tuple[str, Tuple[str, str], Tuple[str, str]], BridgeEdge] = {}
    review: Dict[Tuple[str, Tuple[str, str], Tuple[str, str]], BridgeEdge] = {}

    for i, left in enumerate(mentions):
        if not _is_bridgeable_name(left, min_overlap_chars):
            continue
        for j, right in enumerate(mentions[i + 1 :], start=i + 1):
            if left["kg_id"] == right["kg_id"] or not _is_bridgeable_name(right, min_overlap_chars):
                continue
            semantic_similarity = semantic_scores.get((i, j))
            score, method, signals = _score_candidate(
                left,
                right,
                semantic_similarity=semantic_similarity,
            )
            if method == "incompatible_regulation_letter":
                continue
            strong_prefilter = (
                method in {"normalized_name_exact", "shared_legal_citation", "alias_overlap"}
                or signals["name_similarity"] >= 0.86
                or signals["name_token_overlap"] >= 0.55
                or signals["neighborhood_token_overlap"] >= 0.25
                or ((i, j) in semantic_allowed)
            )
            if not strong_prefilter or score < review_threshold:
                continue
            bridge_type = _choose_bridge_type(method, set(signals.get("shared_citations", [])))
            tier = "tier1_deterministic" if method in {"normalized_name_exact", "shared_legal_citation", "alias_overlap"} else "tier2_scored"
            edge = _make_edge(
                left=left,
                right=right,
                bridge_type=bridge_type,
                confidence=score,
                method=method,
                signals=signals,
                tier=tier if score >= accept_threshold else "tier3_review",
                stamp=stamp,
            )
            if score >= accept_threshold:
                _put_best_edge(accepted, edge)
                review.pop(_canonical_edge_key(edge), None)
            elif _canonical_edge_key(edge) not in accepted:
                _put_best_edge(review, edge)

    accepted_edges = sorted(accepted.values(), key=lambda edge: edge.confidence, reverse=True)
    review_edges = sorted(review.values(), key=lambda edge: edge.confidence, reverse=True)
    if verbose_logging:
        method_counts = defaultdict(int)
        for edge in accepted_edges:
            method_counts[edge.provenance["method"]] += 1
        review_counts = defaultdict(int)
        for edge in review_edges:
            review_counts[edge.provenance["method"]] += 1
        logger.info(
            "Bridge scan complete: accepted=%s review=%s",
            len(accepted_edges),
            len(review_edges),
        )
        for method, count in sorted(method_counts.items()):
            logger.info("Accepted bridge method count: %s=%s", method, count)
        for method, count in sorted(review_counts.items()):
            logger.info("Review bridge method count: %s=%s", method, count)
    return BridgeBuildResult(accepted=accepted_edges, review=review_edges)


def build_candidate_bridges(
    kg_to_entity_file: Dict[str, Path],
    min_overlap_chars: int = 6,
    min_confidence: float = 0.78,
    semantic: bool = False,
    semantic_threshold: float = 0.86,
    max_semantic_edges_per_entity: int = 3,
    verbose_logging: bool = False,
) -> List[BridgeEdge]:
    return build_bridge_sets(
        kg_to_entity_file,
        min_overlap_chars=min_overlap_chars,
        accept_threshold=min_confidence,
        semantic_threshold=semantic_threshold,
        max_semantic_edges_per_entity=max_semantic_edges_per_entity,
        semantic=semantic,
        verbose_logging=verbose_logging,
    ).accepted


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-KG bridge edges")
    parser.add_argument("--kg-map", required=True, help="JSON path: {kg_id: entity_jsonl_path}")
    parser.add_argument("--output", required=True, help="Output JSON file for bridge edges")
    parser.add_argument("--review-output", default=None, help="Output JSON file for review candidates")
    parser.add_argument("--min-confidence", type=float, default=0.78)
    parser.add_argument("--review-threshold", type=float, default=0.58)
    parser.add_argument("--semantic", action="store_true", help="Use embedding similarity for semantic bridges")
    parser.add_argument("--semantic-threshold", type=float, default=0.86)
    parser.add_argument("--max-semantic-edges-per-entity", type=int, default=3)
    parser.add_argument("--replace", action="store_true", help="Replace output instead of appending to it")
    parser.add_argument("--verbose-logging", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose_logging else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    kg_map = json.loads(Path(args.kg_map).read_text(encoding="utf-8"))
    kg_to_entity_file = {kg: Path(path) for kg, path in kg_map.items()}
    result = build_bridge_sets(
        kg_to_entity_file,
        accept_threshold=args.min_confidence,
        review_threshold=args.review_threshold,
        semantic=args.semantic,
        semantic_threshold=args.semantic_threshold,
        max_semantic_edges_per_entity=args.max_semantic_edges_per_entity,
        verbose_logging=args.verbose_logging,
    )
    store = BridgeStore(args.output)
    if args.replace:
        store.save(result.accepted)
    else:
        store.add_many(result.accepted)
    review_output = Path(args.review_output) if args.review_output else Path(args.output).with_name(f"{Path(args.output).stem}_review.json")
    BridgeStore(str(review_output)).save(result.review)
    print(f"Saved {len(store.load())} bridge edges to {args.output}")
    print(f"Saved {len(result.review)} review bridge candidates to {review_output}")


if __name__ == "__main__":
    main()
