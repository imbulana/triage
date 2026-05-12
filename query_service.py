import os
import contextvars
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv

from llm_settings import load_llm_settings
from trace_events import current_recorder

load_dotenv()

with open("config.yaml", "r", encoding="utf-8") as _config_file:
    CONFIG = yaml.safe_load(_config_file) or {}


def _model_param(name: str, default: int) -> int:
    try:
        return int(CONFIG.get("model_params", {}).get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _text_for_token_estimate(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _estimate_tokens(value) -> int:
    text = _text_for_token_estimate(value)
    if not text:
        return 0
    wordish = len(re.findall(r"\S+", text))
    charish = math.ceil(len(text) / 3.0)
    return max(charish, math.ceil(wordish * 1.35))


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
        kwargs.setdefault("max_tokens", _model_param("agent_response_max_tokens", 900))
        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        return response.choices[0].message.content or ""

    class _OpenAIManager:
        generate_text = staticmethod(_openai_generate_text)
        available = True

    return _OpenAIManager()


class QueryService:
    def __init__(
        self,
        registry_path: str = "configs/kg_registry.yaml",
        use_llm_func=None,
        require_leanrag: Optional[bool] = None,
    ):
        self.registry = _load_registry(registry_path)
        self.require_leanrag = _env_bool("TRIAGE_REQUIRE_LEANRAG", True) if require_leanrag is None else require_leanrag
        self.instance_manager = None
        self._raw_use_llm_func = use_llm_func
        self.trace_recorder = None
        self.llm_available = use_llm_func is not None
        if self._raw_use_llm_func is None:
            self.instance_manager = _default_llm()
            self._raw_use_llm_func = self.instance_manager.generate_text
            self.llm_available = bool(getattr(self.instance_manager, "available", True))
        self.use_llm_func = self._generate_text

    def _generate_text(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        trace_name = kwargs.pop("__trace_name", "llm.chat_completion")
        trace_metadata = kwargs.pop("__trace_metadata", {}) or {}
        kwargs.setdefault("max_tokens", _model_param("agent_response_max_tokens", 900))
        recorder = current_recorder() or self.trace_recorder
        settings = load_llm_settings()
        model_name = settings["chat_model"]
        context_budget = self._context_budget(prompt, system_prompt, history_messages, kwargs, model_name)
        model_parameters = {
            key: value
            for key, value in kwargs.items()
            if key in {"temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"}
        }
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict):
            json_schema = response_format.get("json_schema") or {}
            trace_metadata = {
                **trace_metadata,
                "response_format": response_format.get("type"),
                "schema_name": json_schema.get("name") or trace_metadata.get("schema_name"),
            }
        trace_metadata = {**trace_metadata, "context_budget": context_budget}
        if recorder and recorder.enabled:
            with recorder.generation(
                trace_name,
                input_value=recorder.llm_input(prompt, system_prompt, history_messages),
                metadata={
                    **trace_metadata,
                    "prompt_chars": len(str(prompt or "")),
                    "system_prompt_chars": len(str(system_prompt or "")),
                },
                model=model_name,
                model_parameters=model_parameters,
            ) as generation:
                self._raise_if_context_exceeded(context_budget)
                raw = self._raw_use_llm_func(prompt, system_prompt=system_prompt, history_messages=history_messages, **kwargs)
                output = recorder.llm_output(raw)
                generation.update(
                    output=output,
                    metadata={"output_chars": len(str(raw or ""))},
                    usage_details={
                        "input_chars": len(str(prompt or "")) + len(str(system_prompt or "")),
                        "output_chars": len(str(raw or "")),
                        "input_tokens_estimate": context_budget["input_tokens_estimate"],
                        "output_tokens_estimate": _estimate_tokens(raw),
                        "max_output_tokens": context_budget["max_output_tokens"],
                    },
                )
                return raw
        self._raise_if_context_exceeded(context_budget)
        return self._raw_use_llm_func(prompt, system_prompt=system_prompt, history_messages=history_messages, **kwargs)

    def _context_budget(self, prompt, system_prompt, history_messages, kwargs, model_name: str) -> Dict[str, int | str | bool]:
        response_format = kwargs.get("response_format")
        prompt_tokens = _estimate_tokens(prompt)
        system_tokens = _estimate_tokens(system_prompt)
        history_tokens = _estimate_tokens(history_messages or [])
        response_format_tokens = _estimate_tokens(response_format)
        input_tokens = prompt_tokens + system_tokens + history_tokens + response_format_tokens
        max_output_tokens = _as_int(kwargs.get("max_tokens"), _model_param("agent_response_max_tokens", 900))
        context_window_tokens = _model_param("max_token_size", 8192)
        total_tokens = input_tokens + max_output_tokens
        return {
            "model": model_name,
            "context_window_tokens": context_window_tokens,
            "prompt_tokens_estimate": prompt_tokens,
            "system_prompt_tokens_estimate": system_tokens,
            "history_tokens_estimate": history_tokens,
            "response_format_tokens_estimate": response_format_tokens,
            "input_tokens_estimate": input_tokens,
            "max_output_tokens": max_output_tokens,
            "total_tokens_estimate": total_tokens,
            "fits_context": total_tokens <= context_window_tokens,
        }

    def _raise_if_context_exceeded(self, budget: Dict[str, int | str | bool]) -> None:
        if budget["fits_context"]:
            return
        raise RuntimeError(
            "LLM context window exceeded for "
            f"{budget['model']}: estimated input tokens "
            f"{budget['input_tokens_estimate']} + max output tokens "
            f"{budget['max_output_tokens']} = {budget['total_tokens_estimate']}, "
            f"but configured context window is {budget['context_window_tokens']}."
        )

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
                response = self._normalize_retrieval_response(response)
                return {
                    "kg_id": kg_id,
                    "context": context,
                    "response": response,
                    "entities": self._parse_entities_from_context(context),
                    "retrieval_mode": "leanrag",
                }
            except Exception as exc:
                if self.require_leanrag:
                    raise RuntimeError(f"LeanRAG retrieval failed for {kg_id}: {exc}") from exc
                fallback = self._query_local(entry, query, topk_override)
                fallback["retrieval_error"] = str(exc)
                return fallback

        if self.require_leanrag:
            raise RuntimeError(f"LeanRAG retrieval unavailable for {kg_id}: {self._leanrag_unavailable_reason(entry)}")
        return self._query_local(entry, query, topk_override)

    def _should_use_leanrag(self, entry: KGRegistryEntry) -> bool:
        return (
            self.llm_available
            and (Path(entry.working_dir) / "milvus_demo.db").exists()
        )

    def _leanrag_unavailable_reason(self, entry: KGRegistryEntry) -> str:
        reasons = []
        if not self.llm_available:
            reasons.append("LLM backend is unavailable")
        if not (Path(entry.working_dir) / "milvus_demo.db").exists():
            reasons.append(f"missing Milvus Lite index: {Path(entry.working_dir) / 'milvus_demo.db'}")
        return "; ".join(reasons) if reasons else "unknown LeanRAG precondition failure"

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
        response = self._normalize_retrieval_response(response)
        return {
            "kg_id": entry.kg_id,
            "context": context,
            "response": response,
            "entities": [row.get("entity_name", "") for row in ranked_entities if row.get("entity_name")],
            "retrieval_mode": "local",
        }

    def _normalize_retrieval_response(self, response: str) -> str:
        text = " ".join(str(response or "").split())
        text = self._flatten_jsonish_response(text)
        text = self._strip_markdown_table(text)
        text = re.sub(r"(?i)^relationship:\s*\[[^\]]+\]\s*evidence:\s*", "", text).strip()
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        alpha_count = sum(1 for char in text if char.isalpha())
        if alpha_count < 5:
            return "No task-relevant KG evidence found."
        return text

    def _flatten_jsonish_response(self, text: str) -> str:
        if not text or text[0] not in "[{":
            return text
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return text
        flattened = self._flatten_json_value(value)
        return " ".join(part for part in flattened if part)

    def _flatten_json_value(self, value) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                parts.extend(self._flatten_json_value(item))
            return parts
        if isinstance(value, dict):
            parts = []
            for key in ["statement", "evidence", "finding", "summary", "label", "value"]:
                if key in value:
                    parts.extend(self._flatten_json_value(value[key]))
            if not parts:
                for key, item in value.items():
                    if str(key).lower() == "source" and str(item).lower().startswith("query provided"):
                        continue
                    parts.extend(self._flatten_json_value(item))
            return parts
        return [str(value)]

    def _strip_markdown_table(self, text: str) -> str:
        if "|" not in text:
            return text
        cells = []
        for row in text.split("|"):
            cell = row.strip()
            if not cell or set(cell) <= {"-", ":"}:
                continue
            cells.append(cell)
        if len(cells) < 3:
            return text
        return "; ".join(cells)

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
            stripped = line.strip()
            if in_entities and "\t\t" in line and not stripped.startswith("entity_name"):
                name = stripped.split("\t\t", 1)[0].strip()
                if name:
                    names.append(name)
        return names

    def query_many(self, kg_ids: List[str], query: str) -> List[Dict]:
        results = []
        with ThreadPoolExecutor(max_workers=max(1, len(kg_ids))) as ex:
            futures = {
                ex.submit(contextvars.copy_context().run, self.query_kg, kg, query): kg
                for kg in kg_ids
            }
            for future in as_completed(futures):
                kg = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    if self.require_leanrag:
                        raise RuntimeError(f"LeanRAG retrieval failed for {kg}: {exc}") from exc
                    results.append({"kg_id": kg, "error": str(exc), "retrieval_mode": "error"})
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
