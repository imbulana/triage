from typing import Any, Dict, List, Optional, Set

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
        query_service: Optional[QueryService] = None,
    ):
        with open(policy_path, "r", encoding="utf-8") as f:
            self.policy = yaml.safe_load(f)["policy"]
        self.query_service = query_service or QueryService(registry_path=registry_path)
        self.domain_agent = DomainAgent(self.query_service)
        self.compliance_agent = ComplianceAgent(
            self.query_service,
            high_risk_threshold=float(self.policy["compliance_high_risk_threshold"]),
        )
        self.routing_agent = RoutingAgent(self.query_service)
        self.resolution_agent = ResolutionAgent(self.query_service)
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

    def _selected_kg_ids(self, selected: Dict[str, List[str]]) -> Set[str]:
        return {kg_id for kg_ids in selected.values() for kg_id in kg_ids}

    def _complaint_text(self, complaint: Dict[str, Any]) -> str:
        field_labels = [
            ("Product", complaint.get("product")),
            ("Sub-product", complaint.get("sub_product")),
            ("Issue", complaint.get("issue")),
            ("Sub-issue", complaint.get("sub_issue")),
            ("Company", complaint.get("company")),
            ("Narrative", complaint.get("narrative")),
        ]
        return "\n".join(f"{label}: {value}" for label, value in field_labels if value)

    def _evidence_entities(self, *agent_results: Dict[str, Any]) -> Set[str]:
        names: Set[str] = set()
        for result in agent_results:
            for row in result.get("evidence", []):
                for entity in row.get("entities", []):
                    if entity:
                        names.add(str(entity).lower())
        return names

    def _relevant_bridge_hops(
        self,
        selected_kg_ids: Set[str],
        evidence_entities: Set[str],
        complaint_text: str,
    ) -> List[Dict[str, Any]]:
        complaint_lower = complaint_text.lower()
        hops = []
        max_hops = int(self.policy.get("max_secondary_kgs", 2))
        for edge in self.bridge_store.load():
            touches_selected_kg = edge.from_kg in selected_kg_ids or edge.to_kg in selected_kg_ids
            from_node = edge.from_node.lower()
            to_node = edge.to_node.lower()
            touches_evidence = from_node in evidence_entities or to_node in evidence_entities
            mentioned = from_node in complaint_lower or to_node in complaint_lower
            if touches_selected_kg and (touches_evidence or mentioned):
                hops.append(edge.__dict__)
            if len(hops) >= max_hops:
                break
        return hops

    def _validate_decision_trace(self, trace: Dict[str, Any]) -> List[str]:
        errors = []
        required = [
            "agent_evidence",
            "bridge_hops",
            "policy_checks",
            "confidence",
            "uncertainty",
            "escalation_reason",
        ]
        for key in required:
            if key not in trace:
                errors.append(f"missing:{key}")
        if not isinstance(trace.get("agent_evidence"), dict):
            errors.append("agent_evidence must be an object")
        if not isinstance(trace.get("bridge_hops"), list):
            errors.append("bridge_hops must be an array")
        if not isinstance(trace.get("policy_checks"), list):
            errors.append("policy_checks must be an array")
        for key in ("confidence", "uncertainty"):
            value = trace.get(key)
            if not isinstance(value, (int, float)) or not (0 <= float(value) <= 1):
                errors.append(f"{key} must be a number between 0 and 1")
        return errors

    def run(self, complaint: Dict[str, Any]) -> Dict[str, Any]:
        complaint_text = self._complaint_text(complaint)
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
            "agent_decisions": {
                "domain": {
                    "source": domain["classification"].get("source"),
                    "confidence": domain["classification"].get("confidence"),
                    "rationale": domain["classification"].get("rationale"),
                    "error": domain.get("classification_error"),
                },
                "compliance": {
                    "source": compliance.get("source"),
                    "confidence": compliance.get("confidence"),
                    "rationale": compliance.get("rationale"),
                    "error": compliance.get("assessment_error"),
                },
                "routing": {
                    "source": routing.get("source"),
                    "confidence": routing.get("confidence"),
                    "rationale": routing.get("rationale"),
                    "error": routing.get("route_error"),
                },
                "resolution": {
                    "source": resolution.get("source"),
                    "confidence": resolution.get("confidence"),
                    "rationale": resolution.get("rationale"),
                    "error": resolution.get("plan_error"),
                },
            },
            "bridge_hops": self._relevant_bridge_hops(
                self._selected_kg_ids(selected),
                self._evidence_entities(domain, compliance, routing, resolution),
                complaint_text,
            ),
            "policy_checks": compliance["policy_checks"],
            "confidence": sum(confidences) / len(confidences),
            "uncertainty": uncertainty,
            "escalation_reason": escalation_reason,
        }
        trace_errors = self._validate_decision_trace(decision_trace)
        if trace_errors:
            decision_trace["trace_errors"] = trace_errors
            if self.policy.get("fail_closed_on_trace_errors", True):
                escalation_reason = "trace_validation_error"
                decision_trace["escalation_reason"] = escalation_reason

        return {
            "classification": domain["classification"],
            "severity": (
                "high"
                if compliance["compliance_risk"] >= float(self.policy["compliance_high_risk_threshold"])
                else "medium"
            ),
            "compliance_risk": compliance["compliance_risk"],
            "route": routing["route"],
            "resolution_plan": resolution["resolution_plan"],
            "customer_response": resolution["resolution_plan"]["customer_response"],
            "decision_trace": decision_trace,
            "escalate": escalation_reason is not None,
        }
