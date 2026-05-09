import json
from typing import Dict, List, Optional, Tuple

from agents.llm_utils import evidence_summary
from agents.schemas import RoutingDecision
from agents.structured_output import generate_pydantic, repair_pydantic
from query_service import QueryService


ROUTES = {"card_ops", "lending_ops", "digital_banking", "credit_reporting_ops", "triage_ops"}


class RoutingAgent:
    def __init__(self, query_service: QueryService):
        self.query_service = query_service

    def run(self, kg_ids: List[str], complaint_text: str, product_hint: str = "unknown") -> Dict:
        evidence = self.query_service.query_many(kg_ids, complaint_text)
        fallback = self._rule_route(complaint_text, product_hint, evidence)
        route_decision, route_error = self._model_route(complaint_text, product_hint, evidence, fallback)
        result = {
            "agent": "routing",
            "evidence": evidence,
            **route_decision,
        }
        if route_error:
            result["route_error"] = route_error
        return result

    def _rule_route(self, complaint_text: str, product_hint: str, evidence: List[Dict]) -> Dict:
        product = f"{product_hint} {complaint_text}".lower()
        if "credit_card" in product or "credit card" in product or "card" in product:
            route = "card_ops"
        elif "loan" in product or "mortgage" in product or "escrow" in product:
            route = "lending_ops"
        elif "banking" in product or "bank" in product or "account" in product or "transfer" in product:
            route = "digital_banking"
        elif "credit_reporting" in product or "credit report" in product:
            route = "credit_reporting_ops"
        else:
            route = "triage_ops"
        return {
            "confidence": 0.7,
            "route": route,
            "source": "rules",
            "rationale": f"Rule route selected from product hint '{product_hint}'.",
        }

    def _model_route(
        self,
        complaint_text: str,
        product_hint: str,
        evidence: List[Dict],
        fallback: Dict,
    ) -> Tuple[Dict, Optional[str]]:
        prompt = _routing_prompt(complaint_text, product_hint, evidence, fallback)
        system_prompt = "You are a CFPB complaint routing agent. Return only JSON matching the provided schema."
        parsed, error, raw = generate_pydantic(
            self.query_service,
            model=RoutingDecision,
            schema_name="RoutingDecision",
            prompt=prompt,
            system_prompt=system_prompt,
        )
        if error:
            if error == "llm_unavailable":
                return fallback, error
            repaired, repair_error = repair_pydantic(
                self.query_service,
                model=RoutingDecision,
                schema_name="RoutingDecision",
                raw=raw,
                original_prompt=prompt,
            )
            if repaired is None:
                return fallback, error if error == "llm_unavailable" else repair_error or error
            parsed = repaired
        return {**parsed.model_dump(), "source": "llm"}, None


def _routing_prompt(complaint_text: str, product_hint: str, evidence: List[Dict], fallback: Dict) -> str:
    return "\n".join(
        [
            "Choose the best owner team for this complaint.",
            "",
            f"Allowed routes: {', '.join(sorted(ROUTES))}",
            "Return JSON with exactly these keys:",
            '{"route": "...", "confidence": 0.0-1.0, "rationale": "..."}',
            "",
            f"Product hint: {product_hint}",
            f"Rule fallback suggestion: {json.dumps(fallback, ensure_ascii=False)}",
            "",
            "KG evidence summary:",
            evidence_summary(evidence),
            "",
            "Complaint:",
            complaint_text[:6000],
        ]
    )
