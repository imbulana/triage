from typing import Any, Dict, List, Optional, Set

import yaml

from agents.compliance_agent import ComplianceAgent
from agents.domain_agent import DomainAgent
from agents.resolution_agent import ResolutionAgent
from agents.routing_agent import RoutingAgent
from bridges.bridge_store import BridgeStore
from kg_selection import KGSelector
from query_service import QueryService
from trace_events import (
    TraceRecorder,
    summarize_agent_result,
    summarize_final_result,
    summarize_kg_selection,
)


class ComplaintOrchestrator:
    def __init__(
        self,
        registry_path: str = "configs/kg_registry.yaml",
        policy_path: str = "configs/policy.yaml",
        bridge_store_path: str = "bridges/bridge_edges.json",
        query_service: Optional[QueryService] = None,
        require_leanrag: Optional[bool] = None,
    ):
        with open(policy_path, "r", encoding="utf-8") as f:
            self.policy = yaml.safe_load(f)["policy"]
        self.require_leanrag = (
            bool(self.policy.get("require_leanrag_retrieval", True))
            if require_leanrag is None
            else require_leanrag
        )
        self.query_service = query_service or QueryService(registry_path=registry_path, require_leanrag=self.require_leanrag)
        agent_kwargs = {
            "allow_fallbacks": bool(self.policy.get("allow_agent_fallbacks", False)),
            "allow_structured_repair": bool(self.policy.get("allow_structured_repair", False)),
            "allow_empty_response_retry": bool(self.policy.get("allow_empty_response_retry", False)),
        }
        self.domain_agent = DomainAgent(self.query_service, **agent_kwargs)
        self.compliance_agent = ComplianceAgent(
            self.query_service,
            high_risk_threshold=float(self.policy["compliance_high_risk_threshold"]),
            **agent_kwargs,
        )
        self.routing_agent = RoutingAgent(self.query_service, **agent_kwargs)
        self.resolution_agent = ResolutionAgent(self.query_service, **agent_kwargs)
        self.bridge_store = BridgeStore(bridge_store_path)
        self.kg_selector = KGSelector(
            self.query_service.registry,
            community_probe_required=self.require_leanrag,
            bridge_edges=self.bridge_store.load,
        )

    def _select_kgs(self, complaint_text: str) -> Dict[str, Any]:
        return self.kg_selector.select(
            complaint_text,
            max_domain_kgs=int(self.policy.get("max_parallel_agents", 3)),
        )

    def _uncertainty(self, confidences: List[float]) -> float:
        if not confidences:
            return 1.0
        avg = sum(confidences) / len(confidences)
        return max(0.0, min(1.0, 1.0 - avg))

    def _selected_kg_ids(self, selected: Dict[str, Any]) -> Set[str]:
        kg_ids = set()
        for values in selected.values():
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str):
                    kg_ids.add(value)
        return kg_ids

    def _complaint_text(self, complaint: Dict[str, Any]) -> str:
        narrative = complaint.get("narrative") or complaint.get("complaint_narrative") or complaint.get("text") or ""
        return str(narrative).strip()

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

    def run(self, complaint: Dict[str, Any], trace: Optional[TraceRecorder] = None) -> Dict[str, Any]:
        trace = trace or TraceRecorder()
        complaint_text = self._complaint_text(complaint)
        root_input = {
            "complaint_id": complaint.get("complaint_id"),
            "narrative_chars": len(complaint_text),
            "require_leanrag": self.require_leanrag,
        }
        previous_trace_recorder = getattr(self.query_service, "trace_recorder", None)
        self.query_service.trace_recorder = trace
        with trace.span("orchestrator.run", input_value=root_input) as root_span:
            try:
                with trace.span("kg_selection", input_value={"narrative_chars": len(complaint_text)}) as span:
                    selected = self._select_kgs(complaint_text)
                    span.update(output=summarize_kg_selection(selected))

                with trace.span("agent.domain", input_value={"kg_ids": selected["domain_kgs"]}) as span:
                    domain = self.domain_agent.run(selected["domain_kgs"], complaint_text)
                    span.update(output=summarize_agent_result("domain", domain))

                with trace.span("agent.compliance", input_value={"kg_ids": selected["compliance_kgs"]}) as span:
                    compliance = self.compliance_agent.run(selected["compliance_kgs"], complaint_text)
                    span.update(output=summarize_agent_result("compliance", compliance))

                with trace.span(
                    "agent.routing",
                    input_value={
                        "kg_ids": selected["routing_kgs"],
                        "product_hint": domain["classification"].get("product", "unknown"),
                    },
                ) as span:
                    routing = self.routing_agent.run(
                        selected["routing_kgs"],
                        complaint_text,
                        product_hint=domain["classification"].get("product", "unknown"),
                    )
                    span.update(output=summarize_agent_result("routing", routing))

                with trace.span(
                    "agent.resolution",
                    input_value={"kg_ids": selected["domain_kgs"], "route": routing["route"]},
                ) as span:
                    resolution = self.resolution_agent.run(selected["domain_kgs"], complaint_text, route=routing["route"])
                    span.update(output=summarize_agent_result("resolution", resolution))

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
                            "guardrail": routing.get("route_guardrail"),
                        },
                        "resolution": {
                            "source": resolution.get("source"),
                            "confidence": resolution.get("confidence"),
                            "rationale": resolution.get("rationale"),
                            "error": resolution.get("plan_error"),
                        },
                    },
                    "kg_selection": selected,
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

                result = {
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
                root_span.update(output=summarize_final_result(result))
                return result
            finally:
                self.query_service.trace_recorder = previous_trace_recorder
