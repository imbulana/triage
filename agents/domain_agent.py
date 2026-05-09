import json
from typing import Dict, List, Optional, Tuple

from agents.llm_utils import evidence_summary
from agents.schemas import DomainClassification
from agents.structured_output import generate_pydantic, repair_pydantic
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
    def __init__(self, query_service: QueryService):
        self.query_service = query_service

    def run(self, kg_ids: List[str], complaint_text: str) -> Dict:
        evidence = self.query_service.query_many(kg_ids, complaint_text)
        metadata = _extract_metadata(complaint_text)
        fallback, product_hits, issue_hits = self._rule_classification(complaint_text)
        classification, classification_error = self._model_classification(complaint_text, evidence, fallback)
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

    def _model_classification(self, complaint_text: str, evidence: List[Dict], fallback: Dict) -> Tuple[Dict, Optional[str]]:
        prompt = _classification_prompt(complaint_text, evidence, fallback)
        system_prompt = (
            "You classify CFPB complaints for a financial-services triage system. "
            "Return only JSON matching the provided schema."
        )
        parsed, error, raw = generate_pydantic(
            self.query_service,
            model=DomainClassification,
            schema_name="DomainClassification",
            prompt=prompt,
            system_prompt=system_prompt,
        )
        if error:
            if error == "llm_unavailable":
                return fallback, error
            repaired, repair_error = repair_pydantic(
                self.query_service,
                model=DomainClassification,
                schema_name="DomainClassification",
                raw=raw,
                original_prompt=prompt,
            )
            if repaired is None:
                return fallback, repair_error or error
            parsed = repaired
        return {**parsed.model_dump(), "source": "llm"}, None


def _classification_prompt(complaint_text: str, evidence: List[Dict], fallback: Dict) -> str:
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
            "Example: if the complaint says Product: Checking or savings account, then cfpb_product must be exactly 'Checking or savings account', not 'banking'.",
            "Example: if the complaint says Issue: Managing an account, then cfpb_issue must be exactly 'Managing an account', not 'transaction_or_transfer_error'.",
            "For product and issue, map those CFPB values to the allowed internal labels.",
            "Use the complaint metadata/narrative as primary evidence. Use KG evidence only to break ties or map to taxonomy terms.",
            f"Rule fallback suggestion: {json.dumps(fallback, ensure_ascii=False)}",
            "",
            "KG evidence summary:",
            evidence_summary(evidence),
            "",
            "Complaint:",
            complaint_text[:6000],
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
