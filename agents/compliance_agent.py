from typing import Dict, List

from query_service import QueryService


RISK_TERMS = [
    "fraud",
    "identity theft",
    "unfair",
    "discrimination",
    "harassment",
    "error resolution",
    "unauthorized",
]


class ComplianceAgent:
    def __init__(self, query_service: QueryService):
        self.query_service = query_service

    def run(self, kg_ids: List[str], complaint_text: str) -> Dict:
        evidence = self.query_service.query_many(kg_ids, complaint_text)
        lowered = complaint_text.lower()
        hit_count = sum(1 for term in RISK_TERMS if term in lowered)
        compliance_risk = min(1.0, 0.25 + 0.15 * hit_count)
        return {
            "agent": "compliance",
            "evidence": evidence,
            "confidence": min(0.6 + 0.05 * hit_count, 0.95),
            "compliance_risk": compliance_risk,
            "veto": compliance_risk >= 0.7,
            "policy_checks": [
                "consumer complaint process",
                "response timeliness",
                "fair lending/fair servicing"
            ],
        }
