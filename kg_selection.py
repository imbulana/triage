import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from llm_settings import load_llm_settings


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


DOMAIN_KG_PROFILES = {
    "kg_banking_domain": {
        "description": (
            "Deposit accounts, checking and savings accounts, bank account access, ACH and wire transfers, "
            "debit cards, ATM transactions, deposits, withdrawals, overdrafts, payment transfers, and Regulation E."
        ),
        "phrases": {
            "checking account": 4.0,
            "savings account": 4.0,
            "bank account": 4.0,
            "debit card": 3.0,
            "atm card": 3.0,
            "cash deposit": 3.0,
            "wire transfer": 3.0,
            "ach transfer": 3.0,
            "account transfer": 3.0,
            "mobile wallet": 2.5,
            "overdraft fee": 2.5,
        },
        "terms": {
            "bank": 2.0,
            "account": 1.5,
            "checking": 2.5,
            "savings": 2.5,
            "deposit": 2.0,
            "withdrawal": 2.0,
            "transfer": 2.0,
            "ach": 2.5,
            "overdraft": 2.5,
            "debit": 1.5,
            "atm": 1.5,
            "funds": 1.0,
        },
    },
    "kg_credit_domain": {
        "description": (
            "Credit cards, charge cards, credit reporting, credit scores, credit bureaus, consumer reports, "
            "billing disputes, inquiries, furnishers, tradelines, and Fair Credit Reporting Act concerns."
        ),
        "phrases": {
            "credit card": 4.0,
            "credit report": 4.0,
            "credit score": 4.0,
            "credit bureau": 4.0,
            "consumer report": 3.5,
            "hard inquiry": 3.0,
            "identity theft": 2.5,
            "billing dispute": 2.5,
        },
        "terms": {
            "credit": 2.5,
            "card": 1.5,
            "report": 1.5,
            "score": 1.5,
            "bureau": 2.0,
            "equifax": 2.5,
            "experian": 2.5,
            "transunion": 2.5,
            "inquiry": 2.0,
            "furnisher": 2.0,
            "tradeline": 2.0,
        },
    },
    "kg_lending_domain": {
        "description": (
            "Loans and lending products including mortgages, student loans, auto or vehicle loans, payday loans, "
            "title loans, loan servicing, escrow, foreclosure, repossession, and refinancing."
        ),
        "phrases": {
            "student loan": 4.0,
            "mortgage loan": 4.0,
            "personal loan": 3.5,
            "auto loan": 3.5,
            "vehicle loan": 3.5,
            "payday loan": 3.5,
            "loan servicing": 3.0,
            "escrow account": 3.0,
            "loan modification": 3.0,
        },
        "terms": {
            "loan": 2.5,
            "mortgage": 3.0,
            "student": 2.0,
            "servicer": 2.0,
            "servicing": 2.0,
            "escrow": 2.5,
            "foreclosure": 2.5,
            "refinance": 2.0,
            "repossession": 2.0,
            "title": 1.0,
            "payday": 2.0,
        },
    },
}


class KGSelector:
    def __init__(
        self,
        registry: Dict[str, Any],
        embed_texts: Optional[Callable[[List[str]], List[List[float]]]] = None,
        semantic_enabled: bool = True,
        community_probe_enabled: bool = True,
        community_probe_required: bool = False,
        bridge_edges: Optional[Callable[[], Iterable[Any]] | Iterable[Any]] = None,
    ):
        self.registry = registry
        self.embed_texts = embed_texts or self._embed_with_configured_backend
        self.semantic_enabled = semantic_enabled
        self.community_probe_enabled = community_probe_enabled
        self.community_probe_required = community_probe_required
        self.bridge_edges = bridge_edges
        self.semantic_disabled = False
        self.community_probe_disabled = False
        self.embedding_cache: Dict[str, List[float]] = {}
        self.kg_centroid_cache: Dict[str, Optional[List[float]]] = {}

    def select(self, narrative: str, max_domain_kgs: int = 3) -> Dict[str, Any]:
        trajectory = self.community_trajectory(narrative)
        if trajectory.get("available"):
            return self._select_from_trajectory(narrative, trajectory, max_domain_kgs)
        if self.community_probe_required:
            raise RuntimeError(f"Community-probe KG selection is required but unavailable: {trajectory.get('reason')}")

        scores = self.score_domain_kgs(narrative)
        domain_kgs = [kg_id for kg_id, score in scores if score >= 2.0][:max_domain_kgs]

        available = self.available_kgs()
        if "kg_complaints_core" in available and (
            not domain_kgs or len(domain_kgs) < max_domain_kgs
        ):
            domain_kgs.append("kg_complaints_core")
        if not domain_kgs:
            domain_kgs = [kg_id for kg_id in ["kg_complaints_core", *available] if kg_id in available][:1]
        domain_kgs = _unique_in_order(domain_kgs)

        return {
            "domain_kgs": domain_kgs[:max_domain_kgs],
            "compliance_kgs": [kg_id for kg_id in ["kg_regulatory_policy"] if kg_id in available],
            "routing_kgs": [kg_id for kg_id in ["kg_regulatory_policy", "kg_complaints_core"] if kg_id in available],
            "domain_kg_scores": [
                {"kg_id": kg_id, "score": score}
                for kg_id, score in scores
                if score > 0
            ],
            "method": "semantic_weighted" if self.semantic_enabled and not self.semantic_disabled else "weighted",
            "community_probe": trajectory,
        }

    def _select_from_trajectory(
        self,
        narrative: str,
        trajectory: Dict[str, Any],
        max_domain_kgs: int,
    ) -> Dict[str, Any]:
        available = self.available_kgs()
        lexical_scores = dict(self.score_domain_kgs(narrative))
        community_scores = {
            row["kg_id"]: float(row["score"])
            for row in trajectory.get("kg_scores", [])
        }
        bridge_bonus = {
            kg_id: 0.0
            for kg_id in community_scores
        }
        for edge in trajectory.get("bridge_trajectory", []):
            confidence = float(edge.get("confidence", 0.0))
            bridge_bonus[edge["from_kg"]] = bridge_bonus.get(edge["from_kg"], 0.0) + confidence * 0.8
            bridge_bonus[edge["to_kg"]] = bridge_bonus.get(edge["to_kg"], 0.0) + confidence * 0.8

        combined = []
        for kg_id, community_score in community_scores.items():
            score = (community_score * 6.0) + (lexical_scores.get(kg_id, 0.0) * 0.35) + bridge_bonus.get(kg_id, 0.0)
            combined.append((kg_id, round(score, 4)))
        combined.sort(key=lambda item: (-item[1], item[0]))

        domain_kgs = [
            kg_id
            for kg_id, score in combined
            if kg_id in DOMAIN_KG_PROFILES or kg_id == "kg_complaints_core"
        ][:max_domain_kgs]
        if "kg_complaints_core" in available and (
            not domain_kgs or len(domain_kgs) < max_domain_kgs
        ):
            domain_kgs.append("kg_complaints_core")
        if not domain_kgs:
            domain_kgs = [kg_id for kg_id in ["kg_complaints_core", *available] if kg_id in available][:1]
        domain_kgs = _unique_in_order(domain_kgs)

        return {
            "domain_kgs": domain_kgs[:max_domain_kgs],
            "compliance_kgs": [kg_id for kg_id in ["kg_regulatory_policy"] if kg_id in available],
            "routing_kgs": [kg_id for kg_id in ["kg_regulatory_policy", "kg_complaints_core"] if kg_id in available],
            "domain_kg_scores": [
                {"kg_id": kg_id, "score": score}
                for kg_id, score in combined
                if score > 0
            ],
            "method": "community_probe_trajectory",
            "primary_kg": combined[0][0] if combined else None,
            "supporting_kgs": [kg_id for kg_id, _ in combined[1:max_domain_kgs]],
            "community_probe": trajectory,
        }

    def available_kgs(self) -> Set[str]:
        return {kg_id for kg_id, entry in self.registry.items() if getattr(entry, "enabled", True)}

    def score_domain_kgs(self, narrative: str) -> List[Tuple[str, float]]:
        text = narrative.lower()
        tokens = set(re.findall(r"[a-z0-9]+", text))
        available = self.available_kgs()
        semantic_scores = self.semantic_domain_scores(narrative)
        scored = []
        for kg_id, profile in DOMAIN_KG_PROFILES.items():
            if kg_id not in available:
                continue
            score = 0.0
            for phrase, weight in profile["phrases"].items():
                if phrase in text:
                    score += weight
            for term, weight in profile["terms"].items():
                if term in tokens:
                    score += weight
            if kg_id == "kg_credit_domain" and "credit" in tokens and {"deposit", "debit", "account"} & tokens:
                score -= 1.0
            semantic = semantic_scores.get(kg_id)
            if semantic is not None:
                score += max(0.0, semantic - 0.2) * 8.0
            scored.append((kg_id, max(0.0, round(score, 4))))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def semantic_domain_scores(self, narrative: str) -> Dict[str, float]:
        if self.semantic_disabled or not self.semantic_enabled or not narrative.strip():
            return {}

        available_profiles = {
            kg_id: profile
            for kg_id, profile in DOMAIN_KG_PROFILES.items()
            if kg_id in self.available_kgs()
        }
        if not available_profiles:
            return {}

        try:
            query_vector = self.embed([narrative])[0]
            scores = {}
            for kg_id, profile in available_profiles.items():
                profile_vector = self.embed([self.kg_profile_text(kg_id, profile)])[0]
                profile_score = _cosine_similarity(query_vector, profile_vector)
                centroid = self.kg_entity_centroid(kg_id)
                centroid_score = _cosine_similarity(query_vector, centroid) if centroid else profile_score
                scores[kg_id] = (0.4 * profile_score) + (0.6 * centroid_score)
            return scores
        except Exception:
            self.semantic_disabled = True
            return {}

    def community_trajectory(self, narrative: str, topk_per_kg: int = 3) -> Dict[str, Any]:
        if self.community_probe_disabled or not self.community_probe_enabled or not narrative.strip():
            return {"available": False, "reason": "disabled"}

        probe_kgs = [
            kg_id
            for kg_id in self.available_kgs()
            if kg_id in DOMAIN_KG_PROFILES or kg_id in {"kg_complaints_core", "kg_regulatory_policy"}
        ]
        if not probe_kgs:
            return {"available": False, "reason": "no_probe_kgs"}

        try:
            query_vector = self.embed([narrative])[0]
        except Exception:
            self.community_probe_disabled = True
            return {"available": False, "reason": "embedding_unavailable"}

        hits_by_kg: Dict[str, List[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(probe_kgs))) as executor:
            futures = {
                executor.submit(self.probe_communities, kg_id, query_vector, topk_per_kg): kg_id
                for kg_id in probe_kgs
            }
            for future in as_completed(futures):
                kg_id = futures[future]
                try:
                    hits = future.result()
                except Exception:
                    hits = []
                if hits:
                    hits_by_kg[kg_id] = hits

        if not hits_by_kg:
            return {"available": False, "reason": "no_community_hits"}

        kg_scores = [
            {
                "kg_id": kg_id,
                "score": round(max(float(hit.get("score", 0.0)) for hit in hits), 4),
                "hit_count": len(hits),
            }
            for kg_id, hits in hits_by_kg.items()
        ]
        kg_scores.sort(key=lambda row: (-row["score"], row["kg_id"]))

        bridge_trajectory = self.bridge_trajectory(hits_by_kg)
        return {
            "available": True,
            "method": "parallel_community_vector_probe",
            "kg_scores": kg_scores,
            "community_hits": [
                hit
                for kg_id in sorted(hits_by_kg)
                for hit in hits_by_kg[kg_id]
            ],
            "bridge_trajectory": bridge_trajectory,
        }

    def probe_communities(self, kg_id: str, query_vector: List[float], topk: int = 3) -> List[Dict[str, Any]]:
        entry = self.registry.get(kg_id)
        if entry is None:
            return []
        index_path = Path(entry.working_dir) / "milvus_demo.db"
        if not index_path.exists():
            return []

        from pymilvus import MilvusClient

        client = MilvusClient(uri=str(index_path))
        results = client.search(
            collection_name="entity_collection",
            data=[query_vector],
            limit=topk,
            params={"metric_type": "IP", "params": {}},
            filter="level > 0",
            output_fields=["entity_name", "description", "parent", "level", "source_id"],
        )
        hits = []
        for item in results[0]:
            try:
                entity = item.get("entity", {})
                score = item.get("distance", item.get("score", 0.0))
            except AttributeError:
                entity = item["entity"]
                score = item.get("distance", item.get("score", 0.0)) if hasattr(item, "get") else item["distance"]
            hits.append(
                {
                    "kg_id": kg_id,
                    "node": entity.get("entity_name", ""),
                    "description": entity.get("description", ""),
                    "parent": entity.get("parent", ""),
                    "level": entity.get("level"),
                    "source_id": entity.get("source_id", ""),
                    "score": float(score),
                }
            )
        return hits

    def bridge_trajectory(self, hits_by_kg: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        edges = self.load_bridge_edges()
        if not edges:
            return []
        hit_kgs = set(hits_by_kg)
        bridge_rows = []
        for edge in edges:
            from_kg = getattr(edge, "from_kg", "")
            to_kg = getattr(edge, "to_kg", "")
            confidence = float(getattr(edge, "confidence", 0.0))
            if confidence < 0.78 or from_kg not in hit_kgs or to_kg not in hit_kgs:
                continue
            from_supported = self._edge_endpoint_supported_by_hits(edge, from_kg, hits_by_kg[from_kg])
            to_supported = self._edge_endpoint_supported_by_hits(edge, to_kg, hits_by_kg[to_kg])
            if not (from_supported or to_supported):
                continue
            bridge_rows.append(
                {
                    "from_kg": from_kg,
                    "from_node": getattr(edge, "from_node", ""),
                    "to_kg": to_kg,
                    "to_node": getattr(edge, "to_node", ""),
                    "bridge_type": getattr(edge, "bridge_type", ""),
                    "confidence": confidence,
                    "support": "community_overlap" if from_supported and to_supported else "kg_pair_with_endpoint_overlap",
                }
            )
        bridge_rows.sort(key=lambda row: (-row["confidence"], row["from_kg"], row["to_kg"]))
        return bridge_rows[:8]

    def load_bridge_edges(self) -> List[Any]:
        if self.bridge_edges is None:
            return []
        if callable(self.bridge_edges):
            try:
                return list(self.bridge_edges())
            except Exception:
                return []
        return list(self.bridge_edges)

    def _edge_endpoint_supported_by_hits(self, edge: Any, kg_id: str, hits: List[Dict[str, Any]]) -> bool:
        endpoint = getattr(edge, "from_node", "") if getattr(edge, "from_kg", "") == kg_id else getattr(edge, "to_node", "")
        endpoint_norm = _normalize_node_text(endpoint)
        if not endpoint_norm:
            return False
        for hit in hits:
            haystack = _normalize_node_text(f"{hit.get('node', '')} {hit.get('description', '')}")
            if endpoint_norm in haystack:
                return True
        return False

    def kg_profile_text(self, kg_id: str, profile: Dict[str, Any]) -> str:
        phrases = " ".join(profile.get("phrases", {}).keys())
        terms = " ".join(profile.get("terms", {}).keys())
        return f"{kg_id}: {profile.get('description', '')} {phrases} {terms}"

    def kg_entity_centroid(self, kg_id: str) -> Optional[List[float]]:
        if kg_id in self.kg_centroid_cache:
            return self.kg_centroid_cache[kg_id]

        entry = self.registry.get(kg_id)
        if entry is None:
            return None
        entity_path = Path(entry.working_dir) / "entity.jsonl"
        if not entity_path.exists():
            self.kg_centroid_cache[kg_id] = None
            return None

        texts = []
        with entity_path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(texts) >= 48:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                name = str(row.get("entity_name", ""))
                description = str(row.get("description", ""))
                if name or description:
                    texts.append(f"{name}: {description}"[:1200])
        if not texts:
            self.kg_centroid_cache[kg_id] = None
            return None

        vectors = self.embed(texts)
        width = len(vectors[0]) if vectors else 0
        if not width:
            self.kg_centroid_cache[kg_id] = None
            return None
        centroid = [sum(vector[i] for vector in vectors) / len(vectors) for i in range(width)]
        self.kg_centroid_cache[kg_id] = centroid
        return centroid

    def embed(self, texts: List[str]) -> List[List[float]]:
        missing = [text for text in texts if text not in self.embedding_cache]
        if missing:
            for text, vector in zip(missing, self.embed_texts(missing)):
                self.embedding_cache[text] = vector
        return [self.embedding_cache[text] for text in texts]

    def _embed_with_configured_backend(self, texts: List[str]) -> List[List[float]]:
        settings = load_llm_settings()
        from openai import OpenAI

        client_kwargs = {"api_key": settings["embedding_api_key"]}
        if settings["embedding_base_url"]:
            client_kwargs["base_url"] = settings["embedding_base_url"]
        client = OpenAI(**client_kwargs)
        response = client.embeddings.create(model=settings["embedding_model"], input=texts)
        return [list(item.embedding) for item in response.data]


def _normalize_node_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def _unique_in_order(values: List[str]) -> List[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
