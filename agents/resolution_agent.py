from typing import Dict, List, Optional, Tuple

import yaml

from agents.errors import AgentExecutionError
from agents.llm_utils import evidence_summary
from agents.retrieval_queries import agent_retrieval_query
from agents.schemas import ResolutionDecision
from agents.structured_output import generate_pydantic, repair_pydantic
from query_service import QueryService

with open("config.yaml", "r", encoding="utf-8") as _config_file:
    CONFIG = yaml.safe_load(_config_file) or {}


def _resolution_max_tokens() -> int:
    try:
        return int(CONFIG.get("model_params", {}).get("resolution_response_max_tokens", 1100))
    except (TypeError, ValueError):
        return 1100


class ResolutionAgent:
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

    def run(self, kg_ids: List[str], complaint_text: str, route: str) -> Dict:
        evidence = self.query_service.query_many(
            kg_ids,
            agent_retrieval_query("resolution", complaint_text, route=route),
        )
        fallback = self._template_plan(complaint_text, route, evidence)
        plan, plan_error = self._model_plan(complaint_text, route, evidence, fallback)
        result = {
            "agent": "resolution",
            "evidence": evidence,
            **plan,
        }
        if plan_error:
            result["plan_error"] = plan_error
        return result

    def _template_plan(self, complaint_text: str, route: str, evidence: List[Dict]) -> Dict:
        lowered = complaint_text.lower()
        steps = [
            "Acknowledge complaint receipt and provide timeline",
            "Review account events and relevant transaction logs",
            "Apply remediation or correction if confirmed",
            "Send compliant customer response with rationale",
            "Record preventive follow-up action",
        ]
        if "unauthorized" in lowered or "fraud" in lowered:
            steps.insert(2, "Preserve fraud indicators and initiate unauthorized-transaction review")
        if any(term in lowered for term in ["transfer", "transaction", "debit", "overdraft", "dispute"]):
            steps.insert(2, "Reconstruct account ledger, transfer state, and dispute reversal timeline")
        if "discrimination" in lowered or "fair" in lowered:
            steps.insert(2, "Escalate for fair-lending or fair-servicing review before customer closure")
        if "credit report" in lowered or "identity theft" in lowered:
            steps.insert(2, "Check credit-reporting dispute obligations and correction workflow")
        response = (
            "We reviewed your complaint and are actively investigating the issue. "
            "We will provide an update within the applicable response window."
        )
        if route != "triage_ops":
            response = (
                "We routed your complaint to the responsible operations team and are actively "
                "investigating the issue. We will provide an update within the applicable response window."
            )
        return {
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
            "source": "rules",
            "rationale": "Template plan selected from route and complaint terms.",
        }

    def _model_plan(
        self,
        complaint_text: str,
        route: str,
        evidence: List[Dict],
        fallback: Dict,
    ) -> Tuple[Dict, Optional[str]]:
        prompt = _resolution_prompt(complaint_text, route, evidence, fallback)
        system_prompt = (
            "You are a CFPB complaint resolution planning agent. Produce an internal handling plan, not legal advice. Return only JSON matching the provided schema."
        )
        parsed, error, raw = generate_pydantic(
            self.query_service,
            model=ResolutionDecision,
            schema_name="ResolutionDecision",
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=_resolution_max_tokens(),
            allow_empty_retry=self.allow_empty_response_retry,
        )
        if error:
            if self.allow_fallbacks and error == "llm_unavailable":
                return fallback, error
            if self.allow_structured_repair:
                repaired, repair_error = repair_pydantic(
                    self.query_service,
                    model=ResolutionDecision,
                    schema_name="ResolutionDecision",
                    raw=raw,
                    original_prompt=prompt,
                    max_tokens=_resolution_max_tokens(),
                )
                if repaired is not None:
                    parsed = repaired
                else:
                    error = repair_error or error
            if error and parsed is None:
                if self.allow_fallbacks:
                    return fallback, error
                raise AgentExecutionError("resolution", error)
        return {**parsed.model_dump(), "source": "llm"}, None


def _resolution_prompt(complaint_text: str, route: str, evidence: List[Dict], _template: Dict) -> str:
    return "\n".join(
        [
            "Create a concise operational resolution plan for this complaint.",
            "",
            "Return JSON with exactly these keys:",
            '{"resolution_plan": {"owner_team": "...", "actions": ["..."], "customer_response": "...", "preventive_recommendations": ["..."]}, "confidence": 0.0-1.0, "rationale": "..."}',
            "",
            f"Owner team route: {route}",
            "Actions should be concrete internal handling steps, not legal advice.",
            "Customer response should be short, empathetic, and non-committal until investigation confirms facts.",
            "Do not tell the consumer to contact regulators, hire counsel, or gather documents; this plan is for the institution handling the complaint.",
            "Keep rationale to one sentence.",
            "KG evidence summary:",
            evidence_summary(evidence),
            "",
            "Complaint:",
            complaint_text[:6000],
        ]
    )
