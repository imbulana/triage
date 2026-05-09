import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv

from llm_settings import load_llm_settings

load_dotenv()


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
    def _fallback_manager(message: str):
        def _fallback_generate_text(prompt, system_prompt=None, history_messages=None, **kwargs):
            return message

        class _FallbackManager:
            generate_text = staticmethod(_fallback_generate_text)
            available = False

        return _FallbackManager()

    settings = load_llm_settings()
    if not settings["api_key"]:
        return _fallback_manager("Retrieval completed with local deterministic fallback; no LLM backend is configured.")

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        return _fallback_manager("Retrieval completed with local deterministic fallback; openai package is not installed.")

    model = settings["chat_model"]
    client_kwargs = {"api_key": settings["api_key"]}
    if settings["base_url"]:
        client_kwargs["base_url"] = settings["base_url"]
    client = OpenAI(**client_kwargs)

    def _openai_generate_text(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        return response.choices[0].message.content or ""

    class _OpenAIManager:
        generate_text = staticmethod(_openai_generate_text)
        available = True

    return _OpenAIManager()


class QueryService:
    def __init__(self, registry_path: str = "configs/kg_registry.yaml", use_llm_func=None):
        self.registry = _load_registry(registry_path)
        self.instance_manager = None
        self.use_llm_func = use_llm_func
        self.llm_available = use_llm_func is not None
        if self.use_llm_func is None:
            self.instance_manager = _default_llm()
            self.use_llm_func = self.instance_manager.generate_text
            self.llm_available = bool(getattr(self.instance_manager, "available", True))

    def query_kg(self, kg_id: str, query: str, topk_override: Optional[int] = None) -> Dict:
        if kg_id not in self.registry:
            raise KeyError(f"Unknown kg_id: {kg_id}")
        entry = self.registry[kg_id]
        if not entry.enabled:
            raise ValueError(f"KG is disabled: {kg_id}")

        if self._should_use_leanrag(entry):
            try:
                from query_graph import embedding, query_graph

                global_config = {
                    "chunks_file": entry.chunks_file,
                    "embeddings_func": embedding,
                    "working_dir": entry.working_dir,
                    "topk": topk_override if topk_override is not None else entry.topk,
                    "level_mode": entry.level_mode,
                    "use_llm_func": self.use_llm_func,
                }
                context, response = query_graph(global_config=global_config, db=None, query=query)
                return {
                    "kg_id": kg_id,
                    "context": context,
                    "response": response,
                    "entities": self._parse_entities_from_context(context),
                    "retrieval_mode": "leanrag",
                }
            except Exception as exc:
                fallback = self._query_local(entry, query, topk_override)
                fallback["retrieval_error"] = str(exc)
                return fallback

        return self._query_local(entry, query, topk_override)

    def _should_use_leanrag(self, entry: KGRegistryEntry) -> bool:
        return (
            self.llm_available
            and (Path(entry.working_dir) / "milvus_demo.db").exists()
        )

    def _query_local(self, entry: KGRegistryEntry, query: str, topk_override: Optional[int] = None) -> Dict:
        topk = topk_override if topk_override is not None else entry.topk
        query_terms = _terms(query)
        entities = self._read_entities(entry.working_dir)
        chunks = self._read_chunks(entry.chunks_file)

        ranked_entities = sorted(
            entities,
            key=lambda row: _score(query_terms, " ".join([row.get("entity_name", ""), row.get("description", "")])),
            reverse=True,
        )
        ranked_entities = [row for row in ranked_entities if _score(query_terms, row.get("entity_name", "") + " " + row.get("description", "")) > 0][:topk]

        ranked_chunks = sorted(chunks, key=lambda row: _score(query_terms, row.get("text", "")), reverse=True)
        ranked_chunks = [row for row in ranked_chunks if _score(query_terms, row.get("text", "")) > 0][:5]

        context = self._format_local_context(ranked_entities, ranked_chunks)
        try:
            response = self.use_llm_func(query, system_prompt=f"Use this retrieved context:\n{context}")
        except Exception as exc:
            response = (
                "Retrieved local KG evidence, but no LLM response was generated "
                f"because the configured model backend was unavailable: {exc}"
            )
        return {
            "kg_id": entry.kg_id,
            "context": context,
            "response": response,
            "entities": [row.get("entity_name", "") for row in ranked_entities if row.get("entity_name")],
            "retrieval_mode": "local",
        }

    def _read_entities(self, working_dir: str) -> List[Dict]:
        path = Path(working_dir) / "entity.jsonl"
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _read_chunks(self, chunks_file: str) -> List[Dict]:
        path = Path(chunks_file)
        if not path.exists():
            return []
        rows = json.loads(path.read_text(encoding="utf-8"))
        return rows if isinstance(rows, list) else []

    def _format_local_context(self, entities: List[Dict], chunks: List[Dict]) -> str:
        entity_lines = ["entity_name\t\tdescription\t\tsource_id"]
        entity_lines.extend(
            f"{row.get('entity_name', '')}\t\t{row.get('description', '')}\t\t{row.get('source_id', '')}"
            for row in entities
        )
        chunk_lines = [row.get("text", "")[:1200] for row in chunks]
        return "\n".join(
            [
                "entity_information:",
                "\n".join(entity_lines),
                "text_units:",
                "\n---\n".join(chunk_lines),
            ]
        )

    def _parse_entities_from_context(self, context: str) -> List[str]:
        names = []
        in_entities = False
        for line in context.splitlines():
            if line.strip().startswith("entity_information"):
                in_entities = True
                continue
            if in_entities and line.strip().endswith(":"):
                break
            if in_entities and "\t\t" in line and not line.startswith("entity_name"):
                name = line.split("\t\t", 1)[0].strip()
                if name:
                    names.append(name)
        return names

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


def _terms(text: str) -> set:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in {"the", "and", "for", "with", "that", "this", "from"}
    }


def _score(query_terms: set, text: str) -> int:
    if not query_terms:
        return 0
    text_terms = _terms(text)
    return len(query_terms & text_terms)
