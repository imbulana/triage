import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class KGRegistryEntry:
    kg_id: str
    working_dir: str
    chunks_file: str
    topk: int
    level_mode: int
    enabled: bool


def _load_registry(registry_path: str) -> Dict[str, KGRegistryEntry]:
    data = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
    registry = {}
    for row in data.get("kgs", []):
        entry = KGRegistryEntry(**row)
        registry[entry.kg_id] = entry
    return registry


def _default_llm():
    try:
        from tools.utils import InstanceManager
    except Exception:
        def _fallback_generate_text(prompt, system_prompt=None, history_messages=None, **kwargs):
            return "LLM backend not configured."

        class _FallbackManager:
            generate_text = staticmethod(_fallback_generate_text)

        return _FallbackManager()

    num = int(os.getenv("LEANRAG_LLM_INSTANCES", "1"))
    base_url = os.getenv("LEANRAG_LLM_BASE_URL", "http://localhost")
    port = int(os.getenv("LEANRAG_LLM_PORT", "8001"))
    model = os.getenv("LEANRAG_LLM_MODEL", "qwen2.5")
    return InstanceManager(
        url=base_url,
        ports=[port for _ in range(num)],
        gpus=[i for i in range(num)],
        generate_model=model,
        startup_delay=0,
    )


class QueryService:
    def __init__(self, registry_path: str = "configs/kg_registry.yaml", use_llm_func=None):
        self.registry = _load_registry(registry_path)
        self.instance_manager = None
        self.use_llm_func = use_llm_func
        if self.use_llm_func is None:
            self.instance_manager = _default_llm()
            self.use_llm_func = self.instance_manager.generate_text

    def query_kg(self, kg_id: str, query: str, topk_override: Optional[int] = None) -> Dict:
        from query_graph import embedding, query_graph

        if kg_id not in self.registry:
            raise KeyError(f"Unknown kg_id: {kg_id}")
        entry = self.registry[kg_id]
        if not entry.enabled:
            raise ValueError(f"KG is disabled: {kg_id}")

        global_config = {
            "chunks_file": entry.chunks_file,
            "embeddings_func": embedding,
            "working_dir": entry.working_dir,
            "topk": topk_override if topk_override is not None else entry.topk,
            "level_mode": entry.level_mode,
            "use_llm_func": self.use_llm_func,
        }
        context, response = query_graph(global_config=global_config, db=None, query=query)
        return {"kg_id": kg_id, "context": context, "response": response}

    def query_many(self, kg_ids: List[str], query: str) -> List[Dict]:
        results = []
        with ThreadPoolExecutor(max_workers=max(1, len(kg_ids))) as ex:
            futures = {ex.submit(self.query_kg, kg, query): kg for kg in kg_ids}
            for future in as_completed(futures):
                kg = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({"kg_id": kg, "error": str(exc)})
        return results
