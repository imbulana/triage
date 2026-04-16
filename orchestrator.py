from typing import Any, Dict, List

import yaml

from agents.compliance_agent import ComplianceAgent
from agents.domain_agent import DomainAgent
from agents.resolution_agent import ResolutionAgent
from agents.routing_agent import RoutingAgent
from bridges.bridge_store import BridgeStore
from query_service import QueryService


class ComplaintOrchestrator:
    def __init__(
        self,
        registry_path: str = "configs/kg_registry.yaml",
        policy_path: str = "configs/policy.yaml",
        bridge_store_path: str = "bridges/bridge_edges.json",
    ):
        self.query_service = QueryService(registry_path=registry_path)
        self.domain_agent = DomainAgent(self.query_service)
        self.compliance_agent = ComplianceAgent(self.query_service)
        self.routing_agent = RoutingAgent(self.query_service)
        self.resolution_agent = ResolutionAgent(self.query_service)
        with open(policy_path, "r", encoding="utf-8") as f:
            self.policy = yaml.safe_load(f)["policy"]
        self.bridge_store = BridgeStore(bridge_store_path)

    def _select_kgs(self, complaint_text: str) -> Dict[str, List[str]]:
        t = complaint_text.lower()
        domain_kgs = []
        if any(k in t for k in ["credit", "card", "report"]):
            domain_kgs.append("kg_credit_domain")
        if any(k in t for k in ["loan", "mortgage", "student"]):
            domain_kgs.append("kg_lending_domain")
        if any(k in t for k in ["bank", "account", "transfer", "ach"]):
            domain_kgs.append("kg_banking_domain")
        if not domain_kgs:
            domain_kgs = ["kg_complaints_core"]

        return {
            "domain_kgs": domain_kgs[: self.policy.get("max_parallel_agents", 3)],
            "compliance_kgs": ["kg_regulatory_policy"],
            "routing_kgs": ["kg_regulatory_policy", "kg_complaints_core"],
        }

    def _uncertainty(self, confidences: List[float]) -> float:
        if not confidences:
            return 1.0
        avg = sum(confidences) / len(confidences)
        return max(0.0, min(1.0, 1.0 - avg))

    def run(self, complaint: Dict[str, Any]) -> Dict[str, Any]:
        complaint_text = complaint.get("narrative", "")
        selected = self._select_kgs(complaint_text)

        domain = self.domain_agent.run(selected["domain_kgs"], complaint_text)
        compliance = self.compliance_agent.run(selected["compliance_kgs"], complaint_text)
        routing = self.routing_agent.run(
            selected["routing_kgs"], complaint_text, product_hint=domain["classification"].get("product", "unknown")
        )
        resolution = self.resolution_agent.run(selected["domain_kgs"], complaint_text, route=routing["route"])

        confidences = [domain["confidence"], compliance["confidence"], routing["confidence"], resolution["confidence"]]
        uncertainty = self._uncertainty(confidences)

        fairness_metric = float(complaint.get("fairness_parity_gap", 0.0))
        escalation_reason = None
        if compliance["veto"]:
            escalation_reason = "compliance_veto"
        elif uncertainty > float(self.policy["uncertainty_threshold"]):
            escalation_reason = "high_uncertainty"
        elif fairness_metric > float(self.policy["fairness_parity_limit"]):
            escalation_reason = "fairness_guardrail"

        decision_trace = {
            "agent_evidence": {
                "domain": domain["evidence"],
                "compliance": compliance["evidence"],
                "routing": routing["evidence"],
                "resolution": resolution["evidence"],
            },
            "bridge_hops": [e.__dict__ for e in self.bridge_store.load()],
            "policy_checks": compliance["policy_checks"],
            "confidence": sum(confidences) / len(confidences),
            "uncertainty": uncertainty,
            "escalation_reason": escalation_reason,
        }

        return {
            "classification": domain["classification"],
            "severity": "high" if compliance["compliance_risk"] >= 0.7 else "medium",
            "compliance_risk": compliance["compliance_risk"],
            "route": routing["route"],
            "resolution_plan": resolution["resolution_plan"],
            "customer_response": resolution["resolution_plan"]["customer_response"],
            "decision_trace": decision_trace,
            "escalate": escalation_reason is not None,
        }
