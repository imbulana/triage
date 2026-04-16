from typing import Dict, List

from query_service import QueryService


class RoutingAgent:
    def __init__(self, query_service: QueryService):
        self.query_service = query_service

    def run(self, kg_ids: List[str], complaint_text: str, product_hint: str = "unknown") -> Dict:
        evidence = self.query_service.query_many(kg_ids, complaint_text)
        product = product_hint.lower()
        if "credit" in product:
            route = "card_ops"
        elif "loan" in product or "mortgage" in product:
            route = "lending_ops"
        elif "bank" in product or "account" in product:
            route = "digital_banking"
        else:
            route = "triage_ops"
        return {
            "agent": "routing",
            "evidence": evidence,
            "confidence": 0.7,
            "route": route,
        }
