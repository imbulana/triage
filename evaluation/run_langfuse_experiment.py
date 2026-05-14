import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langfuse import Evaluation, get_client

from evaluation.evaluate_orchestrator import evaluate_one, failed_evaluation_row
from orchestrator import ComplaintOrchestrator


LABEL_KEYS = ["product", "sub_product", "issue", "sub_issue"]
EXPECTED_KEYS = [*LABEL_KEYS, "expected_route", "expected_escalate"]


def stable_id(*parts: Any, length: int = 32) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:length]


def load_rows(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected {path} to contain a JSON array.")
    return rows[:limit] if limit else rows


def item_get(item: Any, field: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def experiment_complaint(item: Any) -> Dict[str, Any]:
    input_data = item_get(item, "input", {}) or {}
    expected_output = item_get(item, "expected_output", {}) or {}
    if not isinstance(input_data, dict):
        input_data = {"narrative": str(input_data)}
    if not isinstance(expected_output, dict):
        expected_output = {}

    complaint = dict(input_data)
    for key in EXPECTED_KEYS:
        if key in expected_output and expected_output[key] is not None:
            complaint[key] = expected_output[key]
    return complaint


def compact_evaluation_row(row: Dict[str, Any]) -> Dict[str, Any]:
    result = row.get("result") or {}
    trace = result.get("decision_trace", {}) if isinstance(result, dict) else {}
    compact_result = {
        "classification": result.get("classification"),
        "severity": result.get("severity"),
        "compliance_risk": result.get("compliance_risk"),
        "route": result.get("route"),
        "resolution_plan": result.get("resolution_plan"),
        "customer_response": result.get("customer_response"),
        "escalate": result.get("escalate"),
        "decision_trace": {
            "agent_decisions": trace.get("agent_decisions"),
            "kg_selection": trace.get("kg_selection"),
            "policy_checks": trace.get("policy_checks"),
            "confidence": trace.get("confidence"),
            "uncertainty": trace.get("uncertainty"),
            "escalation_reason": trace.get("escalation_reason"),
        },
    } if result else None
    return {
        "complaint_id": row.get("complaint_id"),
        "evaluation_mode": row.get("evaluation_mode"),
        "error": row.get("error"),
        "taxonomy": row.get("taxonomy"),
        "validation": row.get("validation"),
        "scores": row.get("scores", []),
        "prediction_summary": row.get("prediction_summary"),
        "result": compact_result,
    }


def output_scores_evaluator(*, output: Dict[str, Any], **_: Any) -> List[Evaluation]:
    evaluations: List[Evaluation] = []
    if not isinstance(output, dict):
        return [
            Evaluation(
                name="evaluation_success",
                value=False,
                data_type="BOOLEAN",
                comment="Task output was not a JSON object.",
            )
        ]
    for score in output.get("scores", []):
        value = score.get("value")
        data_type = score.get("data_type") or "NUMERIC"
        if data_type == "BOOLEAN":
            value = bool(value)
        evaluations.append(
            Evaluation(
                name=score["name"],
                value=value,
                data_type=data_type,
                comment=score.get("comment"),
            )
        )
    return evaluations


def aggregate_item_scores(item_results: List[Any]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    for item_result in item_results:
        output = getattr(item_result, "output", {}) or {}
        if not isinstance(output, dict):
            continue
        for score in output.get("scores", []):
            value = score.get("value")
            if isinstance(value, bool):
                numeric = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                numeric = float(value)
            else:
                continue
            values.setdefault(str(score.get("name")), []).append(numeric)
    return {
        name: sum(score_values) / len(score_values)
        for name, score_values in values.items()
        if score_values
    }


def publish_dataset_run_scores(langfuse: Any, dataset_run_id: Optional[str], aggregates: Dict[str, float]) -> int:
    if not dataset_run_id:
        return 0
    emitted = 0
    for name, value in sorted(aggregates.items()):
        langfuse.create_score(
            dataset_run_id=dataset_run_id,
            name=f"run_avg_{name}",
            value=value,
            data_type="NUMERIC",
            score_id=stable_id(dataset_run_id, "run_avg", name),
            comment="Aggregate mean across experiment item outputs.",
        )
        emitted += 1
    return emitted


def create_dataset_items(langfuse: Any, dataset_name: str, rows: List[Dict[str, Any]]) -> None:
    try:
        langfuse.create_dataset(
            name=dataset_name,
            description="Stratified labeled CFPB complaint narratives for complaint triage evaluation.",
            metadata={"source": "data/CFPB/complaints.csv", "row_count": len(rows)},
        )
    except Exception:
        # Dataset may already exist. Item creation below is idempotent via stable ids.
        pass

    for row in rows:
        input_data = {
            "complaint_id": row.get("complaint_id"),
            "narrative": row.get("narrative"),
            "company": row.get("company"),
            "date_received": row.get("date_received"),
            "state": row.get("state"),
            "submitted_via": row.get("submitted_via"),
        }
        expected_output = {key: row.get(key) for key in EXPECTED_KEYS if key in row}
        metadata = {
            "product": row.get("product"),
            "issue": row.get("issue"),
            "sub_product": row.get("sub_product"),
            "sub_issue": row.get("sub_issue"),
            "source": row.get("source"),
        }
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=input_data,
                expected_output=expected_output,
                metadata=metadata,
                id=stable_id(dataset_name, row.get("complaint_id")),
            )
        except Exception:
            # Existing item ids are fine for repeatable runs.
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complaint orchestrator as a Langfuse dataset experiment.")
    parser.add_argument("--input", default="evaluation/inputs/cfpb_90_stratified_labeled.json")
    parser.add_argument("--dataset-name", default="cfpb-90-stratified-labeled")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--experiment-name", default="complaint-triage")
    parser.add_argument("--description", default="Evaluate complaint triage classification, routing, traceability, and resolution quality.")
    parser.add_argument("--registry", default="configs/kg_registry.yaml")
    parser.add_argument("--policy", default="configs/policy.yaml")
    parser.add_argument("--bridges", default="bridges/bridge_edges.json")
    parser.add_argument("--allow-local-fallback", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--output", default="outputs/evals/langfuse_experiment_result.json")
    args = parser.parse_args()

    langfuse = get_client()
    if not langfuse.auth_check():
        raise RuntimeError(
            "Langfuse authentication failed. Set LANGFUSE_PUBLIC_KEY, "
            "LANGFUSE_SECRET_KEY, and LANGFUSE_HOST before running experiments."
        )

    rows = load_rows(Path(args.input), limit=args.limit)
    create_dataset_items(langfuse, args.dataset_name, rows)
    dataset = langfuse.get_dataset(args.dataset_name, fetch_items_page_size=max(50, len(rows)))

    expected_item_ids = {stable_id(args.dataset_name, row.get("complaint_id")) for row in rows}
    data = [item for item in dataset.items if item.id in expected_item_ids]
    if len(data) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} dataset items, found {len(data)} in Langfuse dataset.")

    orchestrator = ComplaintOrchestrator(
        registry_path=args.registry,
        policy_path=args.policy,
        bridge_store_path=args.bridges,
        require_leanrag=not args.allow_local_fallback,
    )

    def task(*, item: Any, **_: Any) -> Dict[str, Any]:
        complaint = experiment_complaint(item)
        try:
            row = evaluate_one(orchestrator, complaint)
        except Exception as exc:
            row = failed_evaluation_row(complaint, exc)
        return compact_evaluation_row(row)

    result = langfuse.run_experiment(
        name=args.experiment_name,
        run_name=args.run_name,
        description=args.description,
        data=data,
        task=task,
        evaluators=[output_scores_evaluator],
        max_concurrency=args.max_concurrency,
        metadata={
            "input": args.input,
            "dataset_name": args.dataset_name,
            "registry": args.registry,
            "bridges": args.bridges,
            "allow_local_fallback": str(bool(args.allow_local_fallback)),
        },
    )
    aggregate_scores = aggregate_item_scores(result.item_results)
    dataset_run_scores_emitted = publish_dataset_run_scores(langfuse, result.dataset_run_id, aggregate_scores)
    langfuse.flush()

    payload = {
        "experiment_name": args.experiment_name,
        "run_name": result.run_name,
        "dataset_run_id": result.dataset_run_id,
        "dataset_run_url": result.dataset_run_url,
        "item_count": len(result.item_results),
        "aggregate_scores": aggregate_scores,
        "dataset_run_scores_emitted": dataset_run_scores_emitted,
        "run_evaluations": [evaluation.__dict__ for evaluation in result.run_evaluations],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
