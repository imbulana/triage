from typing import Dict, List, Optional, Tuple

from agents.errors import AgentExecutionError
from agents.llm_utils import evidence_summary
from agents.retrieval_queries import agent_retrieval_query
from agents.schemas import DomainClassification
from agents.structured_output import generate_pydantic, repair_pydantic
from cfpb_taxonomy import best_taxonomy_path, candidate_taxonomy_paths, format_taxonomy_candidates
from query_service import QueryService


PRODUCT_RULES = [
    ("credit_card", ["credit card", "card", "billing", "charge", "interest"]),
    ("credit_reporting", ["credit report", "credit bureau", "identity theft", "tradeline"]),
    ("mortgage", ["mortgage", "escrow", "foreclosure", "servicer"]),
    ("student_loan", ["student loan", "education loan"]),
    ("banking", ["bank", "account", "deposit", "transfer", "ach", "debit", "overdraft"]),
]

PRODUCT_LABELS = {label for label, _ in PRODUCT_RULES} | {"unknown"}

ISSUE_RULES = [
    ("unauthorized_transaction", ["unauthorized", "fraud", "identity theft"]),
    (
        "transaction_or_transfer_error",
        [
            "transfer",
            "transaction",
            "debit",
            "debited",
            "overdraft",
            "funds not handled",
            "funds not received",
            "wrong amount",
            "wrong day",
        ],
    ),
    ("dispute_handling", ["dispute", "claim reversal", "reversed a claim", "investigation"]),
    ("billing_or_payment_error", ["billing", "payment", "charged", "fee", "interest"]),
    ("servicing_delay", ["delay", "timeline", "response", "servicing"]),
    ("discrimination_or_udaap", ["discrimination", "unfair", "deceptive", "abusive", "harassment"]),
    ("account_opening_or_closure", ["opening an account", "closing an account", "closed account"]),
    ("funds_availability", ["funds not available", "availability", "hold", "withholding"]),
    ("account_access", ["access", "login", "online banking", "mobile access"]),
    ("fee_or_penalty", ["fee", "penalty", "overdraft fee", "non-sufficient funds"]),
]

ISSUE_LABELS = {label for label, _ in ISSUE_RULES} | {"unknown"}


class DomainAgent:
    def __init__(
        self,
        query_service: QueryService,
        *,
        allow_fallbacks: bool = False,
        allow_structured_repair: bool = False,
        allow_empty_response_retry: bool = False,
    ):
        self.query_service = query_service
        self.allow_fallbacks = allow_fallbacks
        self.allow_structured_repair = allow_structured_repair
        self.allow_empty_response_retry = allow_empty_response_retry

    def run(self, kg_ids: List[str], complaint_text: str) -> Dict:
        evidence = self.query_service.query_many(kg_ids, agent_retrieval_query("domain", complaint_text))
        metadata = _extract_metadata(complaint_text)
        heuristic, product_hits, issue_hits = self._rule_classification(complaint_text)
        taxonomy_candidates = candidate_taxonomy_paths(
            complaint_text,
            internal_product=heuristic.get("product", "unknown"),
            internal_issue=heuristic.get("issue", "unknown"),
        )
        classification, classification_error = self._model_classification(
            complaint_text,
            evidence,
            heuristic,
            taxonomy_candidates,
        )
        if self.allow_fallbacks and _should_use_rule_fallback(classification, heuristic, product_hits, issue_hits):
            classification = heuristic
        classification = _apply_taxonomy_candidate(classification, taxonomy_candidates)
        classification = _align_internal_labels(classification, heuristic, product_hits, issue_hits)
        classification = _preserve_cfpb_metadata(classification, metadata)
        successful_retrievals = sum(1 for row in evidence if "error" not in row)
        classification_confidence = float(classification.get("confidence", 0.0))
        score = max(
            classification_confidence,
            0.45 + 0.1 * successful_retrievals + 0.1 * bool(product_hits) + 0.1 * bool(issue_hits),
        )
        result = {
            "agent": "domain",
            "evidence": evidence,
            "confidence": min(score, 0.95),
            "classification": classification,
        }
        if classification_error:
            result["classification_error"] = classification_error
        return result

    def _rule_classification(self, complaint_text: str) -> Tuple[Dict, List[str], List[str]]:
        lowered = complaint_text.lower()
        product, product_hits = _first_match(lowered, PRODUCT_RULES, "unknown")
        issue, issue_hits = _first_match(lowered, ISSUE_RULES, "unknown")
        confidence = 0.45 + 0.15 * bool(product_hits) + 0.15 * bool(issue_hits)
        return (
            {
                "product": product,
                "cfpb_product": "unknown",
                "cfpb_sub_product": "unknown",
                "issue": issue,
                "cfpb_issue": "unknown",
                "sub_issue": None,
                "cfpb_sub_issue": "unknown",
                "confidence": min(confidence, 0.75),
                "source": "rules",
                "rationale": _rule_rationale(product_hits, issue_hits),
            },
            product_hits,
            issue_hits,
        )

    def _model_classification(
        self,
        complaint_text: str,
        evidence: List[Dict],
        fallback: Dict,
        taxonomy_candidates: List[Dict[str, object]],
    ) -> Tuple[Dict, Optional[str]]:
        prompt = _classification_prompt(complaint_text, evidence, fallback, taxonomy_candidates)
        system_prompt = (
            "You classify CFPB complaints for a financial-services triage system. "
            "The complaint narrative is authoritative; use KG evidence only as supporting context. "
            "Return only JSON matching the provided schema."
        )
        parsed, error, raw = generate_pydantic(
            self.query_service,
            model=DomainClassification,
            schema_name="DomainClassification",
            prompt=prompt,
            system_prompt=system_prompt,
            allow_empty_retry=self.allow_empty_response_retry,
        )
        if error:
            if self.allow_fallbacks and error == "llm_unavailable":
                return fallback, error
            if self.allow_structured_repair:
                repaired, repair_error = repair_pydantic(
                    self.query_service,
                    model=DomainClassification,
                    schema_name="DomainClassification",
                    raw=raw,
                    original_prompt=prompt,
                )
                if repaired is not None:
                    return {**repaired.model_dump(), "source": "llm"}, None
                error = repair_error or error
            if self.allow_fallbacks:
                return fallback, error
            raise AgentExecutionError("domain", error)
        return {**parsed.model_dump(), "source": "llm"}, None


def _classification_prompt(
    complaint_text: str,
    evidence: List[Dict],
    _heuristic: Dict,
    taxonomy_candidates: List[Dict[str, object]],
) -> str:
    return "\n".join(
        [
            "Classify the complaint using the allowed labels.",
            "",
            f"Allowed product labels: {', '.join(sorted(PRODUCT_LABELS))}",
            f"Allowed issue labels: {', '.join(sorted(ISSUE_LABELS))}",
            "",
            "Return JSON with exactly these keys:",
            '{"product": "...", "cfpb_product": "...", "cfpb_sub_product": "...", "issue": "...", "cfpb_issue": "...", "sub_issue": null|string, "cfpb_sub_issue": "...", "confidence": 0.0-1.0, "rationale": "..."}',
            "",
            "For cfpb_product, cfpb_sub_product, cfpb_issue, and cfpb_sub_issue, return the exact CFPB taxonomy strings.",
            "If the complaint includes Product, Sub-product, Issue, or Sub-issue metadata, copy those exact strings into the corresponding cfpb_* fields.",
            "If a CFPB taxonomy field is genuinely unavailable, use the string 'unknown' instead of null.",
            "Do not put internal labels such as 'banking', 'checking_account', 'transaction_or_transfer_error', or 'funds_availability' in cfpb_* fields.",
            "When CFPB taxonomy candidates are provided, choose the best matching candidate and copy its product, sub_product, issue, and sub_issue exactly into cfpb_* fields.",
            "Do not return 'unknown' for cfpb_* fields when a candidate plausibly matches the complaint.",
            "If you use a taxonomy candidate, do not claim in the rationale that CFPB fields or candidates were unavailable.",
            "Example: if the complaint says Product: Checking or savings account, then cfpb_product must be exactly 'Checking or savings account', not 'banking'.",
            "Example: if the complaint says Issue: Managing an account, then cfpb_issue must be exactly 'Managing an account', not 'transaction_or_transfer_error'.",
            "For product and issue, map those CFPB values to the allowed internal labels.",
            "Use the complaint narrative as primary evidence. If labeled metadata is explicitly present in this prompt, use it only for the matching cfpb_* fields.",
            "Keep rationale to one sentence focused on the narrative evidence and selected taxonomy path.",
            "CFPB taxonomy candidates:",
            format_taxonomy_candidates(taxonomy_candidates),
            "",
            "Complaint narrative:",
            complaint_text[:6000],
            "",
            "KG evidence summary:",
            evidence_summary(evidence),
        ]
    )


def _rule_rationale(product_hits: List[str], issue_hits: List[str]) -> str:
    parts = []
    if product_hits:
        parts.append(f"product matched terms: {', '.join(product_hits)}")
    if issue_hits:
        parts.append(f"issue matched terms: {', '.join(issue_hits)}")
    return "; ".join(parts) if parts else "no product or issue keywords matched"


def _first_match(text: str, rules, default: str):
    for label, keywords in rules:
        hits = [keyword for keyword in keywords if keyword in text]
        if hits:
            return label, hits
    return default, []


def _extract_metadata(complaint_text: str) -> Dict[str, str]:
    labels = {
        "Product": "cfpb_product",
        "Sub-product": "cfpb_sub_product",
        "Issue": "cfpb_issue",
        "Sub-issue": "cfpb_sub_issue",
    }
    metadata = {}
    for line in complaint_text.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = labels.get(label.strip())
        if key and value.strip():
            metadata[key] = value.strip()
    return metadata


def _preserve_cfpb_metadata(classification: Dict, metadata: Dict[str, str]) -> Dict:
    if not metadata:
        return classification
    preserved = dict(classification)
    preserved.update(metadata)
    return preserved


def _apply_taxonomy_candidate(classification: Dict, candidates: List[Dict[str, object]]) -> Dict:
    candidate = best_taxonomy_path(candidates)
    if not candidate:
        return classification
    candidate_score = float(candidate.get("score", 0.0) or 0.0)
    if candidate_score < 6.0:
        return classification
    selected_score = _selected_taxonomy_score(classification, candidates)
    needs_fill = any(
        str(classification.get(key, "unknown") or "unknown").strip().lower() == "unknown"
        for key in ["cfpb_product", "cfpb_sub_product", "cfpb_issue", "cfpb_sub_issue"]
    )
    should_override = selected_score is None or (candidate_score - selected_score) >= 3.0
    if not needs_fill and not should_override:
        return classification
    filled = dict(classification)
    mapping = {
        "cfpb_product": "cfpb_product",
        "cfpb_sub_product": "cfpb_sub_product",
        "cfpb_issue": "cfpb_issue",
        "cfpb_sub_issue": "cfpb_sub_issue",
    }
    for target, source in mapping.items():
        if should_override or str(filled.get(target, "unknown") or "unknown").strip().lower() == "unknown":
            filled[target] = str(candidate[source])
    taxonomy_path = " / ".join(
        str(candidate[key])
        for key in ["cfpb_product", "cfpb_sub_product", "cfpb_issue", "cfpb_sub_issue"]
    )
    if should_override:
        filled["rationale"] = (
            "CFPB taxonomy fields aligned to the highest-scoring allowed taxonomy candidate: "
            f"{taxonomy_path}."
        )
    else:
        rationale = str(filled.get("rationale", "")).strip()
        note = f"CFPB taxonomy fields filled from the highest-scoring allowed taxonomy candidate: {taxonomy_path}."
        filled["rationale"] = f"{rationale} {note}".strip()
    return filled


def _selected_taxonomy_score(classification: Dict, candidates: List[Dict[str, object]]) -> Optional[float]:
    selected = tuple(
        _normalize_taxonomy_value(classification.get(key))
        for key in ["cfpb_product", "cfpb_sub_product", "cfpb_issue", "cfpb_sub_issue"]
    )
    if any(value in {"", "unknown"} for value in selected):
        return None
    for candidate in candidates:
        row = tuple(
            _normalize_taxonomy_value(candidate.get(key))
            for key in ["cfpb_product", "cfpb_sub_product", "cfpb_issue", "cfpb_sub_issue"]
        )
        if row == selected:
            return float(candidate.get("score", 0.0) or 0.0)
    return None


def _normalize_taxonomy_value(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _align_internal_labels(
    classification: Dict,
    fallback: Dict,
    product_hits: List[str],
    issue_hits: List[str],
) -> Dict:
    aligned = dict(classification)
    notes = []
    if product_hits and fallback.get("product") != "unknown" and aligned.get("product") != fallback.get("product"):
        aligned["product"] = fallback["product"]
        notes.append("internal product aligned with narrative keyword evidence")
    if issue_hits and fallback.get("issue") != "unknown" and aligned.get("issue") != fallback.get("issue"):
        aligned["issue"] = fallback["issue"]
        notes.append("internal issue aligned with narrative keyword evidence")
    if notes:
        rationale = str(aligned.get("rationale", "")).strip()
        aligned["rationale"] = f"{rationale} ({'; '.join(notes)}.)".strip()
    return aligned


def _should_use_rule_fallback(
    classification: Dict,
    fallback: Dict,
    product_hits: List[str],
    issue_hits: List[str],
) -> bool:
    if not product_hits and not issue_hits:
        return False
    confidence = float(classification.get("confidence", 0.0) or 0.0)
    fallback_confidence = float(fallback.get("confidence", 0.0) or 0.0)
    if confidence >= fallback_confidence:
        return False
    if classification.get("product") == "unknown" and fallback.get("product") != "unknown":
        return True
    if classification.get("issue") == "unknown" and fallback.get("issue") != "unknown":
        return True
    rationale = str(classification.get("rationale", "")).lower()
    return "no specific complaint narrative" in rationale or "no complaint narrative" in rationale
