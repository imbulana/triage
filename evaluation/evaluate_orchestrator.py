import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComplaintOrchestrator
from trace_events import TraceRecorder


TAXONOMY_FIELDS = [
    ("product", "cfpb_product"),
    ("sub_product", "cfpb_sub_product"),
    ("issue", "cfpb_issue"),
    ("sub_issue", "cfpb_sub_issue"),
]

DECISION_AGENTS = ["domain", "compliance", "routing", "resolution"]

ROUTE_BY_CFPB_PRODUCT = {
    "checking or savings account": "digital_banking",
    "money transfer, virtual currency, or money service": "digital_banking",
    "credit card": "card_ops",
    "prepaid card": "card_ops",
    "credit reporting or other personal consumer reports": "credit_reporting_ops",
    "debt collection": "triage_ops",
    "debt or credit management": "triage_ops",
    "mortgage": "lending_ops",
    "student loan": "lending_ops",
    "vehicle loan or lease": "lending_ops",
    "payday loan, title loan, personal loan, or advance loan": "lending_ops",
}

SCORE_COMMENT_PREFIX = "triage evaluation"


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def exact_match(expected: Any, actual: Any) -> bool:
    return normalize_label(expected) == normalize_label(actual)


def model_input(complaint: Dict[str, Any]) -> Dict[str, Any]:
    hidden = dict(complaint)
    for key in ["product", "sub_product", "issue", "sub_issue"]:
        hidden.pop(key, None)
    return hidden


def boolean_score(value: bool) -> float:
    return 1 if bool(value) else 0


def score_payload(
    name: str,
    value: Any,
    *,
    data_type: str = "NUMERIC",
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "data_type": data_type,
        "comment": comment,
    }


def expected_route(complaint: Dict[str, Any]) -> Optional[str]:
    for key in ["expected_route", "route", "owner_team"]:
        if complaint.get(key):
            return str(complaint[key])
    product = normalize_label(complaint.get("product"))
    return ROUTE_BY_CFPB_PRODUCT.get(product)


def expected_escalation(complaint: Dict[str, Any]) -> Optional[bool]:
    for key in ["expected_escalate", "escalate", "should_escalate"]:
        if key in complaint:
            return bool(complaint[key])
    return None


def resolution_completeness(validation: Dict[str, Any], result: Dict[str, Any]) -> float:
    plan = result.get("resolution_plan", {}) if isinstance(result, dict) else {}
    checks = [
        bool(validation.get("resolution_has_owner_team")),
        int(validation.get("resolution_actions_count") or 0) >= 3,
        bool(validation.get("customer_response_nonempty")),
        bool(plan.get("preventive_recommendations")),
    ]
    return mean(boolean_score(check) for check in checks)


def build_scores(
    complaint: Dict[str, Any],
    result: Dict[str, Any],
    field_scores: Dict[str, Dict[str, Any]],
    taxonomy_exact_match_accuracy: float,
    all_fields_exact: bool,
    validation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    route_expected = expected_route(complaint)
    escalation_expected = expected_escalation(complaint)
    trace = result.get("decision_trace", {})
    scores = [
        score_payload(
            "evaluation_success",
            1.0,
            data_type="BOOLEAN",
            comment="The orchestrator completed and produced an evaluable result.",
        ),
        score_payload(
            "taxonomy_exact_match_accuracy",
            taxonomy_exact_match_accuracy,
            comment="Mean exact match over CFPB product, sub-product, issue, and sub-issue.",
        ),
        score_payload(
            "taxonomy_all_fields_exact",
            boolean_score(all_fields_exact),
            data_type="BOOLEAN",
            comment="All CFPB taxonomy fields exactly matched the labeled complaint.",
        ),
        score_payload(
            "decision_trace_valid",
            boolean_score(validation["decision_trace_valid"]),
            data_type="BOOLEAN",
            comment="Decision trace includes the required regulator-facing evidence fields.",
        ),
        score_payload(
            "all_agent_decisions_present",
            boolean_score(validation["all_agent_decisions_present"]),
            data_type="BOOLEAN",
            comment="Domain, compliance, routing, and resolution decisions are all present.",
        ),
        score_payload(
            "all_decision_agent_sources_llm",
            boolean_score(validation["all_decision_agent_sources_llm"]),
            data_type="BOOLEAN",
            comment="All specialized agent decisions came from model-backed structured outputs.",
        ),
        score_payload(
            "route_matches_owner_team",
            boolean_score(validation["route_matches_owner_team"]),
            data_type="BOOLEAN",
            comment="The top-level route matches the resolution plan owner team.",
        ),
        score_payload(
            "resolution_completeness",
            resolution_completeness(validation, result),
            comment="Heuristic completeness over owner, actions, customer response, and prevention.",
        ),
        score_payload(
            "customer_response_present",
            boolean_score(validation["customer_response_nonempty"]),
            data_type="BOOLEAN",
            comment="Customer response text is non-empty.",
        ),
    ]
    for field_name, score in field_scores.items():
        scores.append(
            score_payload(
                f"taxonomy_{field_name}_exact_match",
                boolean_score(score["exact_match"]),
                data_type="BOOLEAN",
                comment=f"Expected={score['expected']!r}; predicted={score['predicted']!r}",
            )
        )
    if route_expected:
        scores.append(
            score_payload(
                "route_expected_match",
                boolean_score(result.get("route") == route_expected),
                data_type="BOOLEAN",
                comment=f"Expected route={route_expected!r}; predicted={result.get('route')!r}",
            )
        )
    if escalation_expected is not None:
        scores.append(
            score_payload(
                "escalation_expected_match",
                boolean_score(result.get("escalate") == escalation_expected),
                data_type="BOOLEAN",
                comment=f"Expected escalate={escalation_expected}; predicted={result.get('escalate')}",
            )
        )
    if isinstance(trace.get("confidence"), (int, float)):
        scores.append(score_payload("decision_confidence", float(trace["confidence"])))
    if isinstance(trace.get("uncertainty"), (int, float)):
        scores.append(score_payload("decision_uncertainty", float(trace["uncertainty"])))
    if isinstance(result.get("compliance_risk"), (int, float)):
        scores.append(score_payload("compliance_risk", float(result["compliance_risk"])))
    return scores


def failed_evaluation_row(complaint: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    error = f"{type(exc).__name__}: {exc}"
    scores = [
        score_payload(
            "evaluation_success",
            0.0,
            data_type="BOOLEAN",
            comment=error,
        )
    ]
    return {
        "complaint_id": complaint.get("complaint_id"),
        "evaluation_mode": "narrative_only",
        "error": error,
        "taxonomy": {
            "field_scores": {},
            "exact_match_accuracy": 0.0,
            "all_fields_exact": False,
        },
        "validation": {
            "all_agent_decisions_present": False,
            "all_decision_agent_sources_llm": False,
            "agent_errors": {"evaluation": error},
            "decision_trace_valid": False,
            "customer_response_nonempty": False,
            "resolution_actions_count": 0,
            "resolution_has_owner_team": False,
            "route_matches_owner_team": False,
            "route_expected": expected_route(complaint),
            "route_expected_match": None,
            "expected_escalate": expected_escalation(complaint),
            "escalation_expected_match": None,
            "confidence": None,
            "uncertainty": None,
            "compliance_risk": None,
            "escalate": None,
        },
        "scores": scores,
        "prediction_summary": {
            "classification": {},
            "severity": None,
            "route": None,
            "escalate": None,
            "agent_sources": {},
        },
        "result": None,
    }


def evaluate_one(
    orchestrator: ComplaintOrchestrator,
    complaint: Dict[str, Any],
    trace_recorder: TraceRecorder | None = None,
) -> Dict[str, Any]:
    result = orchestrator.run(model_input(complaint), trace=trace_recorder)
    classification = result.get("classification", {})
    field_scores = {}
    for expected_key, predicted_key in TAXONOMY_FIELDS:
        expected = complaint.get(expected_key)
        predicted = classification.get(predicted_key)
        field_scores[expected_key] = {
            "expected": expected,
            "predicted": predicted,
            "exact_match": exact_match(expected, predicted),
        }

    exact_values = [score["exact_match"] for score in field_scores.values()]
    taxonomy_exact_match_accuracy = mean(exact_values) if exact_values else 0.0
    all_fields_exact = all(exact_values) if exact_values else False
    decision_trace = result.get("decision_trace", {})
    agent_decisions = decision_trace.get("agent_decisions", {})
    agent_sources = {
        agent: decision.get("source")
        for agent, decision in agent_decisions.items()
        if isinstance(decision, dict)
    }
    agent_errors = {
        agent: decision.get("error")
        for agent, decision in agent_decisions.items()
        if isinstance(decision, dict) and decision.get("error")
    }
    resolution_plan = result.get("resolution_plan", {})
    response_text = result.get("customer_response", "")

    validation = {
        "all_agent_decisions_present": all(
            key in agent_decisions for key in DECISION_AGENTS
        ),
        "all_decision_agent_sources_llm": all(
            agent_sources.get(agent) == "llm" for agent in DECISION_AGENTS
        ),
        "agent_errors": agent_errors,
        "decision_trace_valid": "trace_errors" not in decision_trace,
        "customer_response_nonempty": bool(str(response_text).strip()),
        "resolution_actions_count": len(resolution_plan.get("actions", [])),
        "resolution_has_owner_team": bool(resolution_plan.get("owner_team")),
        "route_matches_owner_team": result.get("route") == resolution_plan.get("owner_team"),
        "route_expected": expected_route(complaint),
        "route_expected_match": (
            result.get("route") == expected_route(complaint)
            if expected_route(complaint)
            else None
        ),
        "expected_escalate": expected_escalation(complaint),
        "escalation_expected_match": (
            result.get("escalate") == expected_escalation(complaint)
            if expected_escalation(complaint) is not None
            else None
        ),
        "confidence": decision_trace.get("confidence"),
        "uncertainty": decision_trace.get("uncertainty"),
        "compliance_risk": result.get("compliance_risk"),
        "escalate": result.get("escalate"),
    }
    scores = build_scores(
        complaint,
        result,
        field_scores,
        taxonomy_exact_match_accuracy,
        all_fields_exact,
        validation,
    )

    return {
        "complaint_id": complaint.get("complaint_id"),
        "evaluation_mode": "narrative_only",
        "taxonomy": {
            "field_scores": field_scores,
            "exact_match_accuracy": taxonomy_exact_match_accuracy,
            "all_fields_exact": all_fields_exact,
        },
        "validation": validation,
        "scores": scores,
        "prediction_summary": {
            "classification": classification,
            "severity": result.get("severity"),
            "route": result.get("route"),
            "escalate": result.get("escalate"),
            "agent_sources": agent_sources,
        },
        "result": result,
    }


def load_inputs(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def numeric_score_value(row: Dict[str, Any], name: str) -> Optional[float]:
    for score in row.get("scores", []):
        if score.get("name") == name:
            value = score.get("value")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def mean_available(values: Iterable[Optional[float]]) -> Optional[float]:
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def emit_langfuse_scores(
    trace_recorder: Optional[TraceRecorder],
    row: Dict[str, Any],
    *,
    eval_run_id: str,
) -> int:
    if trace_recorder is None or not trace_recorder.langfuse_trace_id:
        return 0
    emitted = 0
    complaint_id = row.get("complaint_id") or "unknown"
    for score in row.get("scores", []):
        name = str(score["name"])
        comment_parts = [
            SCORE_COMMENT_PREFIX,
            f"eval_run_id={eval_run_id}",
            f"complaint_id={complaint_id}",
        ]
        if score.get("comment"):
            comment_parts.append(str(score["comment"]))
        score_id = hashlib.sha256(f"{trace_recorder.langfuse_trace_id}:{name}".encode("utf-8")).hexdigest()[:32]
        if trace_recorder.score_trace(
            name=name,
            value=score["value"],
            data_type=score.get("data_type") or "NUMERIC",
            comment=" | ".join(comment_parts),
            score_id=score_id,
        ):
            emitted += 1
    return emitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the multi-agent complaint orchestrator.")
    parser.add_argument("--input", required=True, help="Complaint JSON file, or JSON array of complaints")
    parser.add_argument("--registry", default="configs/kg_registry_fast.yaml")
    parser.add_argument("--policy", default="configs/policy.yaml")
    parser.add_argument("--bridges", default="bridges/bridge_edges_fast_semantic.json")
    parser.add_argument("--output", default=None, help="Optional path to write full evaluation JSON")
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Deprecated no-op: labels are hidden from model input by default.",
    )
    parser.add_argument(
        "--allow-local-fallback",
        action="store_true",
        help="Allow entity/chunk fallback when LeanRAG is unavailable.",
    )
    parser.add_argument("--events-dir", default=None, help="Optional directory for per-complaint trace event JSONL files.")
    parser.add_argument("--stream-events", action="store_true", help="Stream compact trace events to stderr as JSONL.")
    parser.add_argument("--langfuse", action="store_true", help="Mirror each complaint run to Langfuse using LANGFUSE_* env vars.")
    parser.add_argument(
        "--langfuse-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --langfuse is enabled, publish evaluation metrics as Langfuse trace scores.",
    )
    parser.add_argument("--eval-run-name", default="complaint-triage-eval", help="Human-readable evaluation run name for trace metadata.")
    parser.add_argument("--eval-run-id", default=None, help="Stable run id for idempotent trace metadata and score comments.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on the first failed complaint instead of recording a failed evaluation row.")
    parser.add_argument("--trace-id-seed", default=None, help="Optional deterministic Langfuse trace-id seed prefix.")
    parser.add_argument(
        "--trace-llm-io",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Capture LLM prompt bodies in traces. Defaults to config.yaml trace.capture_llm_io.",
    )
    parser.add_argument("--trace-llm-max-chars", type=int, default=None, help="Max chars for captured LLM prompt/output bodies.")
    args = parser.parse_args()

    orchestrator = ComplaintOrchestrator(
        registry_path=args.registry,
        policy_path=args.policy,
        bridge_store_path=args.bridges,
        require_leanrag=not args.allow_local_fallback,
    )
    complaints = load_inputs(Path(args.input))
    eval_run_id = args.eval_run_id or uuid.uuid4().hex
    rows = []
    langfuse_scores_emitted = 0
    for index, complaint in enumerate(complaints):
        complaint_id = complaint.get("complaint_id") or f"row_{index + 1}"
        events_path = None
        if args.events_dir:
            events_dir = Path(args.events_dir)
            events_dir.mkdir(parents=True, exist_ok=True)
            safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(complaint_id))
            events_path = str(events_dir / f"{safe_id}.events.jsonl")
        trace = None
        if events_path or args.stream_events or args.langfuse:
            seed_prefix = args.trace_id_seed or eval_run_id
            seed = f"{seed_prefix}:{complaint_id}"
            trace = TraceRecorder(
                events_path=events_path,
                stream_events=args.stream_events,
                langfuse_enabled=args.langfuse,
                trace_id_seed=seed,
                capture_llm_io=args.trace_llm_io,
                llm_io_max_chars=args.trace_llm_max_chars,
                trace_name="complaint-evaluation",
                metadata={
                    "input": args.input,
                    "complaint_id": str(complaint_id),
                    "eval_run_name": args.eval_run_name,
                    "eval_run_id": eval_run_id,
                    "eval_row_index": index,
                },
            )
        try:
            try:
                row = evaluate_one(orchestrator, complaint, trace_recorder=trace)
            except Exception as exc:
                if args.fail_fast:
                    raise
                row = failed_evaluation_row(complaint, exc)
            if args.langfuse and args.langfuse_scores:
                langfuse_scores_emitted += emit_langfuse_scores(
                    trace,
                    row,
                    eval_run_id=eval_run_id,
                )
        finally:
            if trace:
                trace.close()
        if events_path:
            row["trace_events"] = events_path
        rows.append(row)
    summary = {
        "count": len(rows),
        "taxonomy_exact_match_accuracy": mean(row["taxonomy"]["exact_match_accuracy"] for row in rows),
        "all_fields_exact_rate": mean(row["taxonomy"]["all_fields_exact"] for row in rows),
        "all_decision_agent_sources_llm_rate": mean(
            row["validation"]["all_decision_agent_sources_llm"] for row in rows
        ),
        "decision_trace_valid_rate": mean(row["validation"]["decision_trace_valid"] for row in rows),
        "customer_response_nonempty_rate": mean(row["validation"]["customer_response_nonempty"] for row in rows),
        "avg_resolution_actions_count": mean(row["validation"]["resolution_actions_count"] for row in rows),
        "route_expected_match_rate": mean_available(
            numeric_score_value(row, "route_expected_match") for row in rows
        ),
        "escalation_expected_match_rate": mean_available(
            numeric_score_value(row, "escalation_expected_match") for row in rows
        ),
        "avg_resolution_completeness": mean_available(
            numeric_score_value(row, "resolution_completeness") for row in rows
        ),
        "avg_decision_confidence": mean_available(
            numeric_score_value(row, "decision_confidence") for row in rows
        ),
        "avg_decision_uncertainty": mean_available(
            numeric_score_value(row, "decision_uncertainty") for row in rows
        ),
        "avg_compliance_risk": mean_available(
            numeric_score_value(row, "compliance_risk") for row in rows
        ),
        "evaluation_success_rate": mean_available(
            numeric_score_value(row, "evaluation_success") for row in rows
        ),
    }
    payload = {
        "eval_run": {
            "id": eval_run_id,
            "name": args.eval_run_name,
            "langfuse_enabled": bool(args.langfuse),
            "langfuse_scores_enabled": bool(args.langfuse and args.langfuse_scores),
            "langfuse_scores_emitted": langfuse_scores_emitted,
        },
        "summary": summary,
        "evaluations": rows,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
