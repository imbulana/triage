import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from bridges.build_bridges import build_bridge_sets
from bridges.bridge_store import BridgeStore
from ingestion.fetch_sources import fetch_and_store, import_local_xml
from ingestion.extract_triples import extract_triples_for_kg
from ingestion.prepare_chunks import prepare_kg_chunks
from ingestion.simple_kg import bootstrap_kg_from_chunks
from orchestrator import ComplaintOrchestrator
from query_service import QueryService
from trace_events import TraceRecorder
from trace_report import build_trace_report, write_trace_report

load_dotenv()
logger = logging.getLogger(__name__)


def _load_registry(path: str = "configs/kg_registry.yaml") -> Dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cmd_ingest(args) -> None:
    if args.verbose_logging:
        logger.info("Starting ingest with registry=%s raw_out=%s", args.registry, args.raw_out)
    if args.xml_root:
        total = import_local_xml(args.xml_root, args.raw_out, verbose_logging=args.verbose_logging)
        print(f"Imported {total} local XML documents")
    else:
        total = fetch_and_store(args.manifest, args.raw_out, verbose_logging=args.verbose_logging)
        print(f"Fetched {total} source documents")

    registry = _load_registry(args.registry)
    chunked = 0
    for kg in registry.get("kgs", []):
        if not kg.get("enabled", True):
            continue
        if args.verbose_logging:
            logger.info("Preparing chunks for kg=%s", kg["kg_id"])
        count = prepare_kg_chunks(args.raw_out, kg["kg_id"], kg["chunks_file"], verbose_logging=args.verbose_logging)
        print(f"{kg['kg_id']}: {count} chunks")
        chunked += count
    print(f"Prepared total chunks: {chunked}")


def cmd_build_kgs(args) -> None:
    registry = _load_registry(args.registry)
    for kg in registry.get("kgs", []):
        if not kg.get("enabled", True):
            continue
        working_dir = kg["working_dir"]
        if args.verbose_logging:
            logger.info("Starting build for kg=%s working_dir=%s", kg["kg_id"], working_dir)
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        entity_path = Path(working_dir) / "entity.jsonl"
        relation_path = Path(working_dir) / "relation.jsonl"
        if args.bootstrap or not (entity_path.exists() and relation_path.exists()):
            stats = bootstrap_kg_from_chunks(
                kg["chunks_file"],
                working_dir,
                kg["kg_id"],
                verbose_logging=args.verbose_logging,
            )
            print(
                f"[build-kgs] {kg['kg_id']}: bootstrapped {stats['entities']} entities "
                f"and {stats['relations']} relations from {stats['chunks']} chunks."
            )
        else:
            print(f"[build-kgs] {kg['kg_id']}: using existing extracted entity.jsonl/relation.jsonl.")
        if args.run_commands:
            print(f"[build-kgs] Running LeanRAG clustering/index build in {working_dir}.")
            cmd = ["python", "build_graph.py", "-p", working_dir]
            if args.verbose_logging:
                cmd.append("--verbose-logging")
            subprocess.run(cmd, check=True)


def cmd_extract_triples(args) -> None:
    registry = _load_registry(args.registry)
    selected_kgs = [
        kg
        for kg in registry.get("kgs", [])
        if kg.get("enabled", True) and (not args.kg_id or kg["kg_id"] == args.kg_id)
    ]
    for kg in tqdm(selected_kgs, desc="extract KGs", unit="kg", disable=args.no_progress):
        print(f"[extract-triples] {kg['kg_id']}: extracting from {kg['chunks_file']}")
        stats = extract_triples_for_kg(
            kg["chunks_file"],
            kg["working_dir"],
            model=args.model,
            base_url=args.base_url,
            max_concurrency=args.max_concurrency,
            limit=args.limit,
            show_progress=not args.no_progress,
            verbose_logging=args.verbose_logging,
        )
        print(
            f"[extract-triples] {kg['kg_id']}: extracted {stats['entities']} entities "
            f"and {stats['relations']} relations from {stats['chunks']} chunks."
        )


def cmd_build_bridges(args) -> None:
    registry = _load_registry(args.registry)
    kg_to_entity = {}
    for kg in registry.get("kgs", []):
        if kg.get("enabled", True):
            kg_to_entity[kg["kg_id"]] = str(Path(kg["working_dir"]) / "entity.jsonl")

    if args.verbose_logging:
        logger.info("Starting bridge build for %s KGs", len(kg_to_entity))
    result = build_bridge_sets(
        {k: Path(v) for k, v in kg_to_entity.items()},
        accept_threshold=args.min_confidence,
        review_threshold=args.review_threshold,
        semantic=args.semantic,
        semantic_threshold=args.semantic_threshold,
        max_semantic_edges_per_entity=args.max_semantic_edges_per_entity,
        verbose_logging=args.verbose_logging,
    )
    store = BridgeStore(args.output)
    if args.replace:
        store.save(result.accepted)
    else:
        store.add_many(result.accepted)
    review_output = (
        Path(args.review_output)
        if args.review_output
        else Path(args.output).with_name(f"{Path(args.output).stem}_review.json")
    )
    BridgeStore(str(review_output)).save(result.review)
    print(f"Stored {len(store.load())} bridge edges")
    print(f"Stored {len(result.review)} review bridge candidates in {review_output}")


def cmd_query(args) -> None:
    service = QueryService(registry_path=args.registry, require_leanrag=not args.allow_local_fallback)
    result = service.query_kg(args.kg_id, args.query, topk_override=args.topk)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_orchestrate(args) -> None:
    orchestrator = ComplaintOrchestrator(
        registry_path=args.registry,
        policy_path=args.policy,
        bridge_store_path=args.bridges,
        require_leanrag=not args.allow_local_fallback,
    )
    complaint = json.loads(Path(args.input).read_text(encoding="utf-8"))
    trace = TraceRecorder(
        events_path=args.events_output,
        stream_events=args.stream_events,
        langfuse_enabled=args.langfuse,
        trace_id_seed=args.trace_id_seed,
        capture_llm_io=args.trace_llm_io,
        llm_io_max_chars=args.trace_llm_max_chars,
        metadata={
            "registry": args.registry,
            "policy": args.policy,
            "bridges": args.bridges,
            "input": args.input,
        },
    )
    try:
        result = orchestrator.run(complaint, trace=trace)
    finally:
        trace.close()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.trace_html:
        Path(args.trace_html).parent.mkdir(parents=True, exist_ok=True)
        events = trace.events
        if args.events_output:
            events = [
                json.loads(line)
                for line in Path(args.events_output).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        Path(args.trace_html).write_text(build_trace_report(result, events), encoding="utf-8")
    print(text)


def cmd_trace_report(args) -> None:
    write_trace_report(args.result, args.events, args.output)
    print(f"Wrote trace report to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LeanRAG complaint pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--manifest", default="configs/sources_manifest.yaml")
    p_ingest.add_argument("--registry", default="configs/kg_registry.yaml")
    p_ingest.add_argument("--raw-out", default="datasets/raw_sources")
    p_ingest.add_argument("--xml-root", default=None, help="Optional local XML root: <xml-root>/<kg_id>/*.xml")
    p_ingest.add_argument("--verbose-logging", action="store_true", help="Print timestamped ingest logs")
    p_ingest.set_defaults(func=cmd_ingest)

    p_build = sub.add_parser("build-kgs")
    p_build.add_argument("--registry", default="configs/kg_registry.yaml")
    p_build.add_argument("--bootstrap", action="store_true", help="Force deterministic bootstrap before indexing")
    p_build.add_argument("--run-commands", action="store_true")
    p_build.add_argument("--verbose-logging", action="store_true", help="Print timestamped build logs")
    p_build.set_defaults(func=cmd_build_kgs)

    p_extract = sub.add_parser("extract-triples")
    p_extract.add_argument("--registry", default="configs/kg_registry.yaml")
    p_extract.add_argument("--kg-id", default=None, help="Optional single KG to extract")
    p_extract.add_argument("--model", default=None, help="Override chat model")
    p_extract.add_argument("--base-url", default=None, help="Override OpenAI-compatible base URL")
    p_extract.add_argument("--max-concurrency", type=int, default=4)
    p_extract.add_argument("--limit", type=int, default=None, help="Limit chunks for smoke tests")
    p_extract.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p_extract.add_argument("--verbose-logging", action="store_true", help="Print timestamped extraction logs")
    p_extract.set_defaults(func=cmd_extract_triples)

    p_bridges = sub.add_parser("build-bridges")
    p_bridges.add_argument("--registry", default="configs/kg_registry.yaml")
    p_bridges.add_argument("--output", default="bridges/bridge_edges.json")
    p_bridges.add_argument("--review-output", default=None)
    p_bridges.add_argument("--min-confidence", type=float, default=0.78)
    p_bridges.add_argument("--review-threshold", type=float, default=0.58)
    p_bridges.add_argument("--semantic", action="store_true", help="Use embedding similarity for semantic bridges")
    p_bridges.add_argument("--semantic-threshold", type=float, default=0.86)
    p_bridges.add_argument("--max-semantic-edges-per-entity", type=int, default=3)
    p_bridges.add_argument("--replace", action="store_true", help="Replace output instead of appending to it")
    p_bridges.add_argument("--verbose-logging", action="store_true", help="Print timestamped bridge logs")
    p_bridges.set_defaults(func=cmd_build_bridges)

    p_query = sub.add_parser("query")
    p_query.add_argument("--registry", default="configs/kg_registry.yaml")
    p_query.add_argument("--kg-id", required=True)
    p_query.add_argument("--query", required=True)
    p_query.add_argument("--topk", type=int, default=None)
    p_query.add_argument("--allow-local-fallback", action="store_true", help="Allow entity/chunk fallback when LeanRAG is unavailable")
    p_query.set_defaults(func=cmd_query)

    p_orc = sub.add_parser("orchestrate")
    p_orc.add_argument("--registry", default="configs/kg_registry.yaml")
    p_orc.add_argument("--policy", default="configs/policy.yaml")
    p_orc.add_argument("--bridges", default="bridges/bridge_edges.json")
    p_orc.add_argument("--input", required=True, help="JSON complaint file")
    p_orc.add_argument("--output", default=None, help="Optional path to write orchestrator JSON")
    p_orc.add_argument("--events-output", default=None, help="Optional JSONL path for compact trace events")
    p_orc.add_argument("--stream-events", action="store_true", help="Stream compact trace events to stderr as JSONL")
    p_orc.add_argument("--trace-html", default=None, help="Optional HTML decision-trace report path")
    p_orc.add_argument("--langfuse", action="store_true", help="Mirror trace spans to Langfuse using LANGFUSE_* env vars")
    p_orc.add_argument("--trace-id-seed", default=None, help="Optional deterministic Langfuse trace-id seed")
    p_orc.add_argument(
        "--trace-llm-io",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Capture LLM prompt bodies in traces. Defaults to config.yaml trace.capture_llm_io.",
    )
    p_orc.add_argument("--trace-llm-max-chars", type=int, default=None, help="Max chars for captured LLM prompt/output bodies")
    p_orc.add_argument("--allow-local-fallback", action="store_true", help="Allow entity/chunk fallback when LeanRAG is unavailable")
    p_orc.set_defaults(func=cmd_orchestrate)

    p_trace = sub.add_parser("trace-report")
    p_trace.add_argument("--result", required=True, help="Orchestrator result JSON file")
    p_trace.add_argument("--events", default=None, help="Optional JSONL event stream")
    p_trace.add_argument("--output", required=True, help="HTML report output path")
    p_trace.set_defaults(func=cmd_trace_report)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose_logging", False) else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
