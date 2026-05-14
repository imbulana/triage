from typing import Dict, List, Optional, Tuple

from agents.errors import AgentExecutionError
from agents.llm_utils import evidence_summary
from agents.retrieval_queries import agent_retrieval_query
from agents.schemas import CFPBDomainClassification
from agents.structured_output import generate_pydantic, repair_pydantic
from cfpb_taxonomy import INTERNAL_PRODUCT_TO_CFPB_PRODUCTS, cfpb_label_sets
from query_service import QueryService


CFPB_LABEL_SETS = cfpb_label_sets(include_unknown=True)
CFPB_DISPLAY_LABEL_SETS = cfpb_label_sets(include_unknown=False)
CFPB_PRODUCT_TO_INTERNAL = {
    cfpb_product: internal_product
    for internal_product, cfpb_products in INTERNAL_PRODUCT_TO_CFPB_PRODUCTS.items()
    for cfpb_product in cfpb_products
}

PRODUCT_RULES = [
    ("prepaid_card", ["prepaid card", "prepaid"]),
    ("credit_card", ["credit card", "card", "billing", "charge", "interest"]),
    ("credit_reporting", ["credit report", "credit bureau", "identity theft", "tradeline"]),
    ("debt_collection", ["debt collection", "debt collector", "collection agency", "collect a debt", "debt owed"]),
    ("debt_or_credit_management", ["credit repair", "debt settlement", "debt management", "credit counseling"]),
    ("money_transfer", ["money transfer", "wire transfer", "virtual currency", "crypto", "remittance", "money service"]),
    ("mortgage", ["mortgage", "escrow", "foreclosure", "servicer"]),
    ("payday_personal_loan", ["payday loan", "title loan", "personal loan", "installment loan", "advance loan"]),
    ("student_loan", ["student loan", "education loan"]),
    ("vehicle_loan", ["vehicle loan", "auto loan", "car loan", "lease", "repossession"]),
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
        classification, classification_error = self._model_classification(
            complaint_text,
            evidence,
            heuristic,
        )
        if self.allow_fallbacks and _should_use_rule_fallback(classification, heuristic, product_hits, issue_hits):
            classification = heuristic
        classification = _preserve_cfpb_metadata(classification, metadata)
        classification = _derive_internal_aliases(classification, heuristic)
        classification = _align_internal_labels(classification, heuristic, product_hits, issue_hits)
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
    ) -> Tuple[Dict, Optional[str]]:
        prompt = _classification_prompt(complaint_text, evidence, fallback)
        system_prompt = (
            "You classify CFPB complaints for a financial-services triage system. "
            "The complaint narrative is authoritative; use KG evidence only as supporting context. "
            "Return only JSON matching the provided schema."
        )
        parsed, error, raw = generate_pydantic(
            self.query_service,
            model=CFPBDomainClassification,
            schema_name="CFPBDomainClassification",
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
                    model=CFPBDomainClassification,
                    schema_name="CFPBDomainClassification",
                    raw=raw,
                    original_prompt=prompt,
                )
                if repaired is not None:
                    return _classification_from_cfpb(repaired.model_dump(), fallback), None
                error = repair_error or error
            if self.allow_fallbacks:
                return fallback, error
            raise AgentExecutionError("domain", error)
        return _classification_from_cfpb(parsed.model_dump(), fallback), None


def _classification_prompt(
    complaint_text: str,
    evidence: List[Dict],
    _heuristic: Dict,
) -> str:
    return "\n".join(
        [
            "Classify the complaint using the official CFPB taxonomy labels.",
            "",
            "CFPB labels are the primary classification output.",
            (
                "The complete allowed CFPB label sets are supplied in the JSON schema: "
                f"{len(CFPB_DISPLAY_LABEL_SETS['products'])} products, "
                f"{len(CFPB_DISPLAY_LABEL_SETS['sub_products'])} sub-products, "
                f"{len(CFPB_DISPLAY_LABEL_SETS['issues'])} issues, and "
                f"{len(CFPB_DISPLAY_LABEL_SETS['sub_issues'])} sub-issues, "
                "with 'unknown' allowed when a field is genuinely unavailable."
            ),
            "Do not use per-narrative candidate labels; choose from the full schema-constrained CFPB label universe.",
            "",
            "Return JSON with exactly these keys:",
            '{"product": "...", "sub_product": "...", "issue": "...", "sub_issue": "...", "confidence": 0.0-1.0, "rationale": "..."}',
            "",
            "For product, sub_product, issue, and sub_issue, return exact CFPB taxonomy strings from the allowed label sets.",
            "If the complaint includes Product, Sub-product, Issue, or Sub-issue metadata, copy those exact strings into the corresponding fields.",
            "If a CFPB taxonomy field is genuinely unavailable, use the string 'unknown'.",
            "Do not return internal aliases such as 'banking', 'checking_account', 'transaction_or_transfer_error', or 'funds_availability'.",
            "Example: if the complaint says Product: Checking or savings account, then product must be exactly 'Checking or savings account', not 'banking'.",
            "Example: if the complaint says Issue: Managing an account, then issue must be exactly 'Managing an account', not 'transaction_or_transfer_error'.",
            "Use the complaint narrative as primary evidence. If labeled metadata is explicitly present in this prompt, use it only for the matching CFPB taxonomy fields.",
            "Keep rationale to one sentence focused on the narrative evidence and selected CFPB labels.",
            "",
            "Complaint narrative:",
            complaint_text[:6000],
            "",
            "KG evidence summary:",
            evidence_summary(evidence),
        ]
    )


def _classification_from_cfpb(cfpb_classification: Dict, fallback: Dict) -> Dict:
    classification = {
        "product": "unknown",
        "cfpb_product": str(cfpb_classification.get("product") or "unknown"),
        "cfpb_sub_product": str(cfpb_classification.get("sub_product") or "unknown"),
        "issue": "unknown",
        "cfpb_issue": str(cfpb_classification.get("issue") or "unknown"),
        "sub_issue": None,
        "cfpb_sub_issue": str(cfpb_classification.get("sub_issue") or "unknown"),
        "confidence": cfpb_classification.get("confidence", fallback.get("confidence", 0.0)),
        "source": "llm",
        "rationale": cfpb_classification.get("rationale", ""),
    }
    return _derive_internal_aliases(classification, fallback)


def _derive_internal_aliases(classification: Dict, fallback: Dict) -> Dict:
    derived = dict(classification)
    cfpb_product = str(derived.get("cfpb_product") or "unknown")
    product = CFPB_PRODUCT_TO_INTERNAL.get(cfpb_product)
    if product is None:
        product = fallback.get("product") if fallback.get("product") in PRODUCT_LABELS else "unknown"
    derived["product"] = product or "unknown"

    issue_text = " ".join(
        str(derived.get(key) or "")
        for key in ["cfpb_issue", "cfpb_sub_issue"]
    ).lower()
    issue, _hits = _first_match(issue_text, ISSUE_RULES, "unknown")
    if issue == "unknown" and fallback.get("issue") in ISSUE_LABELS:
        issue = fallback.get("issue")
    derived["issue"] = issue or "unknown"
    derived["sub_issue"] = None
    return derived


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
