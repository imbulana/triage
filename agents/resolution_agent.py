from typing import Dict, List

from query_service import QueryService


class ResolutionAgent:
    def __init__(self, query_service: QueryService):
        self.query_service = query_service

    def run(self, kg_ids: List[str], complaint_text: str, route: str) -> Dict:
        evidence = self.query_service.query_many(kg_ids, complaint_text)
        steps = [
            "Acknowledge complaint receipt and provide timeline",
            "Review account events and relevant transaction logs",
            "Apply remediation or correction if confirmed",
            "Send compliant customer response with rationale",
            "Record preventive follow-up action",
        ]
        response = (
            "We reviewed your complaint and are actively investigating the issue. "
            "We will provide an update within the applicable response window."
        )
        return {
            "agent": "resolution",
            "evidence": evidence,
            "confidence": 0.72,
            "resolution_plan": {
                "owner_team": route,
                "actions": steps,
                "customer_response": response,
                "preventive_recommendations": [
                    "Improve frontline issue tagging",
                    "Add rule-based QA checks for similar complaints",
                ],
            },
        }
