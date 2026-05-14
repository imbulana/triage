from typing import Dict, List, Optional, Tuple

from agents.errors import AgentExecutionError
from agents.llm_utils import evidence_summary
from agents.retrieval_queries import agent_retrieval_query
from agents.schemas import RoutingDecision
from agents.structured_output import generate_pydantic, repair_pydantic
from query_service import QueryService


ROUTES = {"card_ops", "lending_ops", "digital_banking", "credit_reporting_ops", "triage_ops"}

PRODUCT_HINT_ROUTES = {
    "banking": "digital_banking",
    "credit_card": "card_ops",
    "credit_reporting": "credit_reporting_ops",
    "debt_collection": "triage_ops",
    "debt_or_credit_management": "triage_ops",
    "money_transfer": "digital_banking",
    "mortgage": "lending_ops",
    "payday_personal_loan": "lending_ops",
    "prepaid_card": "card_ops",
    "student_loan": "lending_ops",
    "vehicle_loan": "lending_ops",
}


class RoutingAgent:
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

    def run(self, kg_ids: List[str], complaint_text: str, product_hint: str = "unknown") -> Dict:
        evidence = self.query_service.query_many(
            kg_ids,
            agent_retrieval_query("routing", complaint_text, product_hint=product_hint),
        )
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
        hint = str(product_hint or "").strip().lower()
        if hint in PRODUCT_HINT_ROUTES:
            route = PRODUCT_HINT_ROUTES[hint]
        else:
            product = f"{product_hint} {complaint_text}".lower()
            route = _route_from_text(product)
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
        system_prompt = "You are a CFPB complaint routing agent. Choose the accountable operations team. Return only JSON matching the provided schema."
        parsed, error, raw = generate_pydantic(
            self.query_service,
            model=RoutingDecision,
            schema_name="RoutingDecision",
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
                    model=RoutingDecision,
                    schema_name="RoutingDecision",
                    raw=raw,
                    original_prompt=prompt,
                )
                if repaired is not None:
                    parsed = repaired
                else:
                    error = repair_error or error
            if error and parsed is None:
                if self.allow_fallbacks:
                    return fallback, error
                raise AgentExecutionError("routing", error)
        decision = {**parsed.model_dump(), "source": "llm"}
        guardrail = _route_product_guardrail(decision, fallback, product_hint, complaint_text)
        if guardrail:
            decision.update(guardrail)
        return decision, None


def _route_from_text(product: str) -> str:
    if "prepaid_card" in product or "prepaid card" in product:
        route = "card_ops"
    elif "credit_card" in product or "credit card" in product or "card" in product:
        route = "card_ops"
    elif (
        "vehicle_loan" in product
        or "payday_personal_loan" in product
        or "student_loan" in product
        or "loan" in product
        or "mortgage" in product
        or "escrow" in product
    ):
        route = "lending_ops"
    elif (
        "money_transfer" in product
        or "banking" in product
        or "bank" in product
        or "account" in product
        or "transfer" in product
    ):
        route = "digital_banking"
    elif "credit_reporting" in product or "credit report" in product:
        route = "credit_reporting_ops"
    elif "debt_collection" in product or "debt_or_credit_management" in product or "debt collection" in product:
        route = "triage_ops"
    else:
        route = "triage_ops"
    return route


def _routing_prompt(complaint_text: str, product_hint: str, evidence: List[Dict], _heuristic: Dict) -> str:
    return "\n".join(
        [
            "Choose the best owner team for this complaint.",
            "",
            f"Allowed routes: {', '.join(sorted(ROUTES))}",
            "Return JSON with exactly these keys:",
            '{"route": "...", "confidence": 0.0-1.0, "rationale": "..."}',
            "",
            f"Product hint: {product_hint}",
            "Route based on the product and operational failure described in the narrative, not on generic regulatory entities.",
            "If product_hint is banking and the complaint is about checking/savings accounts, deposits, transfers, account debits, overdrafts, or payment movement, choose digital_banking.",
            "If product_hint is money_transfer and the complaint is about wire transfers, remittances, virtual currency, or money services, choose digital_banking.",
            "Choose card_ops only when a credit card, prepaid card, charge card, card issuer, or card-network transaction is the central product.",
            "Choose lending_ops when product_hint is mortgage, student_loan, vehicle_loan, or payday_personal_loan.",
            "Choose credit_reporting_ops when product_hint is credit_reporting.",
            "Choose triage_ops when product_hint is debt_collection or debt_or_credit_management because this system has no dedicated collections route.",
            "Do not choose card_ops merely because the narrative mentions account operations or transaction reversal.",
            "Keep rationale to one sentence.",
            "KG evidence summary:",
            evidence_summary(evidence),
            "",
            "Complaint:",
            complaint_text[:6000],
        ]
    )


def _route_product_guardrail(decision: Dict, fallback: Dict, product_hint: str, complaint_text: str) -> Optional[Dict]:
    route = decision.get("route")
    fallback_route = fallback.get("route")
    product = str(product_hint or "").lower()
    narrative = str(complaint_text or "").lower()
    banking_terms = ["checking", "savings", "bank account", "transfer", "deposit", "overdraft", "account debit"]
    card_terms = ["credit card", "prepaid card", "charge card", "card issuer"]
    is_banking = product in {"banking", "money_transfer"} or any(term in narrative for term in banking_terms)
    is_card_specific = any(term in narrative for term in card_terms)
    if is_banking and not is_card_specific and fallback_route == "digital_banking" and route != "digital_banking":
        return {
            "route": "digital_banking",
            "confidence": min(float(decision.get("confidence", 0.7)), 0.82),
            "rationale": (
                "Aligned to the banking product hint and account-transfer narrative; "
                f"model proposed {route}, but this complaint belongs with digital_banking."
            ),
            "route_guardrail": f"product_hint_conflict:{route}->digital_banking",
        }
    return None
