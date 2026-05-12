from typing import Dict, List, Optional, Tuple

from agents.errors import AgentExecutionError
from agents.llm_utils import evidence_summary
from agents.retrieval_queries import agent_retrieval_query
from agents.schemas import ComplianceAssessment
from agents.structured_output import generate_pydantic, repair_pydantic
from query_service import QueryService


HIGH_RISK_TERMS = [
    "fraud",
    "identity theft",
    "unfair",
    "discrimination",
    "harassment",
    "error resolution",
    "unauthorized",
    "not authorized",
]

MODERATE_RISK_TERMS = [
    "debit",
    "debited",
    "overdraft",
    "dispute",
    "claim reversal",
    "wrong amount",
    "wrong day",
]


class ComplianceAgent:
    def __init__(
        self,
        query_service: QueryService,
        high_risk_threshold: float = 0.7,
        *,
        allow_fallbacks: bool = False,
        allow_structured_repair: bool = False,
        allow_empty_response_retry: bool = False,
    ):
        self.query_service = query_service
        self.high_risk_threshold = high_risk_threshold
        self.allow_fallbacks = allow_fallbacks
        self.allow_structured_repair = allow_structured_repair
        self.allow_empty_response_retry = allow_empty_response_retry

    def run(self, kg_ids: List[str], complaint_text: str) -> Dict:
        evidence = self.query_service.query_many(kg_ids, agent_retrieval_query("compliance", complaint_text))
        fallback = self._rule_assessment(complaint_text, evidence)
        assessment, assessment_error = self._model_assessment(complaint_text, evidence, fallback)
        result = {
            "agent": "compliance",
            "evidence": evidence,
            **assessment,
        }
        if assessment_error:
            result["assessment_error"] = assessment_error
        return result

    def _rule_assessment(self, complaint_text: str, evidence: List[Dict]) -> Dict:
        lowered = complaint_text.lower()
        high_hits = [term for term in HIGH_RISK_TERMS if term in lowered]
        moderate_hits = [term for term in MODERATE_RISK_TERMS if term in lowered]
        compliance_risk = min(1.0, 0.25 + 0.2 * len(high_hits) + 0.08 * len(moderate_hits))
        return {
            "confidence": min(0.6 + 0.06 * len(high_hits) + 0.03 * len(moderate_hits), 0.95),
            "compliance_risk": compliance_risk,
            "veto": compliance_risk >= self.high_risk_threshold,
            "policy_checks": [
                "consumer complaint process",
                "response timeliness",
                "fair lending/fair servicing"
            ],
            "source": "rules",
            "rationale": _rule_rationale(high_hits, moderate_hits),
        }

    def _model_assessment(self, complaint_text: str, evidence: List[Dict], fallback: Dict) -> Tuple[Dict, Optional[str]]:
        prompt = _compliance_prompt(complaint_text, evidence, fallback, self.high_risk_threshold)
        system_prompt = (
            "You are a CFPB compliance triage agent. Assess escalation risk from the narrative and KG evidence. Return only JSON matching the provided schema."
        )
        parsed, error, raw = generate_pydantic(
            self.query_service,
            model=ComplianceAssessment,
            schema_name="ComplianceAssessment",
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
                    model=ComplianceAssessment,
                    schema_name="ComplianceAssessment",
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
                raise AgentExecutionError("compliance", error)
        assessment = parsed.model_dump()
        if assessment["compliance_risk"] >= self.high_risk_threshold:
            assessment["veto"] = True
        return {**assessment, "source": "llm"}, None


def _compliance_prompt(complaint_text: str, evidence: List[Dict], _heuristic: Dict, high_risk_threshold: float) -> str:
    return "\n".join(
        [
            "Assess compliance risk for this CFPB complaint.",
            "",
            "Return JSON with exactly these keys:",
            '{"compliance_risk": 0.0-1.0, "veto": true|false, "policy_checks": ["..."], "confidence": 0.0-1.0, "rationale": "..."}',
            "",
            f"Set veto=true only when compliance_risk >= {high_risk_threshold} or the complaint clearly needs mandatory escalation.",
            "Prefer moderate risk for ordinary transaction disputes unless there is unauthorized activity, discrimination, UDAAP, fraud, legal deadline risk, or repeated institutional failure.",
            "Do not give legal advice or customer-facing instructions; this is an internal risk assessment.",
            "Keep rationale to one sentence naming the main risk drivers.",
            "KG evidence summary:",
            evidence_summary(evidence),
            "",
            "Complaint:",
            complaint_text[:6000],
        ]
    )

def _rule_rationale(high_hits: List[str], moderate_hits: List[str]) -> str:
    parts = []
    if high_hits:
        parts.append(f"high-risk terms: {', '.join(high_hits)}")
    if moderate_hits:
        parts.append(f"moderate-risk terms: {', '.join(moderate_hits)}")
    return "; ".join(parts) if parts else "no compliance risk terms matched"
