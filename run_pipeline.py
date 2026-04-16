import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict

import yaml

from bridges.build_bridges import build_candidate_bridges
from bridges.bridge_store import BridgeStore
from ingestion.fetch_sources import fetch_and_store
from ingestion.prepare_chunks import prepare_kg_chunks
from orchestrator import ComplaintOrchestrator
from query_service import QueryService


def _load_registry(path: str = "configs/kg_registry.yaml") -> Dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cmd_ingest(args) -> None:
    total = fetch_and_store(args.manifest, args.raw_out)
    print(f"Fetched {total} source documents")

    registry = _load_registry(args.registry)
    chunked = 0
    for kg in registry.get("kgs", []):
        if not kg.get("enabled", True):
            continue
        count = prepare_kg_chunks(args.raw_out, kg["kg_id"], kg["chunks_file"])
        print(f"{kg['kg_id']}: {count} chunks")
        chunked += count
    print(f"Prepared total chunks: {chunked}")


def cmd_build_kgs(args) -> None:
    registry = _load_registry(args.registry)
    for kg in registry.get("kgs", []):
        if not kg.get("enabled", True):
            continue
        working_dir = kg["working_dir"]
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        print(f"[build-kgs] Build KG in {working_dir} using existing LeanRAG extraction + build scripts.")
        if args.run_commands:
            subprocess.run(["python", "build_graph.py", "-p", working_dir], check=False)


def cmd_build_bridges(args) -> None:
    registry = _load_registry(args.registry)
    kg_to_entity = {}
    for kg in registry.get("kgs", []):
        if kg.get("enabled", True):
            kg_to_entity[kg["kg_id"]] = str(Path(kg["working_dir"]) / "entity.jsonl")

    edges = build_candidate_bridges({k: Path(v) for k, v in kg_to_entity.items()})
    store = BridgeStore(args.output)
    store.add_many(edges)
    print(f"Stored {len(store.load())} bridge edges")


def cmd_query(args) -> None:
    service = QueryService(registry_path=args.registry)
    result = service.query_kg(args.kg_id, args.query, topk_override=args.topk)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_orchestrate(args) -> None:
    orchestrator = ComplaintOrchestrator(
        registry_path=args.registry,
        policy_path=args.policy,
        bridge_store_path=args.bridges,
    )
    complaint = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = orchestrator.run(complaint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="LeanRAG complaint pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--manifest", default="configs/sources_manifest.yaml")
    p_ingest.add_argument("--registry", default="configs/kg_registry.yaml")
    p_ingest.add_argument("--raw-out", default="datasets/raw_sources")
    p_ingest.set_defaults(func=cmd_ingest)

    p_build = sub.add_parser("build-kgs")
    p_build.add_argument("--registry", default="configs/kg_registry.yaml")
    p_build.add_argument("--run-commands", action="store_true")
    p_build.set_defaults(func=cmd_build_kgs)

    p_bridges = sub.add_parser("build-bridges")
    p_bridges.add_argument("--registry", default="configs/kg_registry.yaml")
    p_bridges.add_argument("--output", default="bridges/bridge_edges.json")
    p_bridges.set_defaults(func=cmd_build_bridges)

    p_query = sub.add_parser("query")
    p_query.add_argument("--registry", default="configs/kg_registry.yaml")
    p_query.add_argument("--kg-id", required=True)
    p_query.add_argument("--query", required=True)
    p_query.add_argument("--topk", type=int, default=None)
    p_query.set_defaults(func=cmd_query)

    p_orc = sub.add_parser("orchestrate")
    p_orc.add_argument("--registry", default="configs/kg_registry.yaml")
    p_orc.add_argument("--policy", default="configs/policy.yaml")
    p_orc.add_argument("--bridges", default="bridges/bridge_edges.json")
    p_orc.add_argument("--input", required=True, help="JSON complaint file")
    p_orc.set_defaults(func=cmd_orchestrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
