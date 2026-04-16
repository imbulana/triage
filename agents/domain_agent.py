from typing import Dict, List

from query_service import QueryService


class DomainAgent:
    def __init__(self, query_service: QueryService):
        self.query_service = query_service

    def run(self, kg_ids: List[str], complaint_text: str) -> Dict:
        evidence = self.query_service.query_many(kg_ids, complaint_text)
        score = 0.5 + 0.1 * sum(1 for row in evidence if "error" not in row)
        return {
            "agent": "domain",
            "evidence": evidence,
            "confidence": min(score, 0.95),
            "classification": {
                "product": "unknown",
                "issue": "unknown",
                "sub_issue": None,
            },
        }
