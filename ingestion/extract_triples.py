import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from GraphExtraction._utils import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
)
from llm_settings import load_llm_settings
from prompt import PROMPTS

load_dotenv()
logger = logging.getLogger(__name__)


def extract_triples_for_kg(
    chunks_file: str,
    working_dir: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    max_concurrency: int = 4,
    limit: int | None = None,
    show_progress: bool = True,
    verbose_logging: bool = False,
) -> Dict[str, int]:
    chunks = _load_chunks(chunks_file)
    if limit is not None:
        chunks = dict(list(chunks.items())[:limit])
    output_dir = Path(working_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if verbose_logging:
        logger.info("Loaded %s chunks from %s", len(chunks), chunks_file)
    stats = asyncio.run(
        _extract_triples(
            chunks,
            output_dir,
            model=model,
            base_url=base_url,
            max_concurrency=max_concurrency,
            show_progress=show_progress,
            verbose_logging=verbose_logging,
        )
    )
    return stats


def _load_chunks(chunks_file: str) -> Dict[str, str]:
    rows = json.loads(Path(chunks_file).read_text(encoding="utf-8"))
    return {str(row["hash_code"]): row.get("text", "") for row in rows if row.get("hash_code")}


async def _extract_triples(
    chunks: Dict[str, str],
    output_dir: Path,
    *,
    model: str | None,
    base_url: str | None,
    max_concurrency: int,
    show_progress: bool,
    verbose_logging: bool,
) -> Dict[str, int]:
    settings = load_llm_settings()
    selected_model = model or settings["extraction_model"]
    use_llm = _make_async_llm(model=model, base_url=base_url, max_concurrency=max_concurrency)
    ordered_chunks = list(chunks.items())
    if verbose_logging:
        logger.info(
            "Starting triple extraction: chunks=%s model=%s concurrency=%s output_dir=%s",
            len(ordered_chunks),
            selected_model,
            max_concurrency,
            output_dir,
        )
        logger.info("Beginning entity extraction phase")

    entity_results = await _run_chunk_tasks(
        [
            _extract_entities(chunk_key, text, use_llm)
            for chunk_key, text in ordered_chunks
        ],
        ordered_chunks=ordered_chunks,
        desc="extract entities",
        show_progress=show_progress,
        verbose_logging=verbose_logging,
    )
    context_entities = {
        chunk_key: sorted(result.keys())
        for (chunk_key, _), result in zip(ordered_chunks, entity_results)
    }

    if verbose_logging:
        logger.info("Entity extraction phase complete; beginning relation extraction phase")
    relation_results = await _run_chunk_tasks(
        [
            _extract_relations(chunk_key, text, context_entities.get(chunk_key, []), use_llm)
            for chunk_key, text in ordered_chunks
        ],
        ordered_chunks=ordered_chunks,
        desc="extract relations",
        show_progress=show_progress,
        verbose_logging=verbose_logging,
    )

    entities = _dedupe_entities(entity_results)
    relations = _dedupe_relations(relation_results)
    if verbose_logging:
        logger.info("Writing entity output to %s", output_dir / "entity.jsonl")
    _write_jsonl(entities, output_dir / "entity.jsonl")
    if verbose_logging:
        logger.info("Writing relation output to %s", output_dir / "relation.jsonl")
    _write_jsonl(relations, output_dir / "relation.jsonl")
    if verbose_logging:
        logger.info(
            "Triple extraction complete: chunks=%s entities=%s relations=%s",
            len(chunks),
            len(entities),
            len(relations),
        )
    return {"chunks": len(chunks), "entities": len(entities), "relations": len(relations)}


async def _run_chunk_tasks(tasks, *, ordered_chunks, desc: str, show_progress: bool, verbose_logging: bool):
    async def _indexed(index, task):
        return index, await task

    indexed_tasks = [
        asyncio.create_task(_indexed(index, task))
        for index, task in enumerate(tasks)
    ]
    results = [None] * len(indexed_tasks)
    completed = asyncio.as_completed(indexed_tasks)
    started_at = time.monotonic()
    if show_progress:
        completed = tqdm(completed, total=len(indexed_tasks), desc=desc, unit="chunk")
    for future in completed:
        index, result = await future
        results[index] = result
        if verbose_logging:
            chunk_key, _ = ordered_chunks[index]
            logger.info(
                "%s progress %s/%s chunk=%s records=%s elapsed=%.1fs",
                desc,
                sum(item is not None for item in results),
                len(indexed_tasks),
                chunk_key,
                len(result),
                time.monotonic() - started_at,
            )
    return results


def _make_async_llm(*, model: str | None, base_url: str | None, max_concurrency: int):
    settings = load_llm_settings()
    selected_model = model or settings["extraction_model"]
    selected_base_url = base_url or settings["base_url"]
    if _is_ollama_base_url(selected_base_url):
        return _make_ollama_llm(
            model=selected_model,
            base_url=selected_base_url,
            max_concurrency=max_concurrency,
        )

    client_kwargs = {"api_key": settings["api_key"]}
    if selected_base_url:
        client_kwargs["base_url"] = selected_base_url
    client = AsyncOpenAI(**client_kwargs)
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def generate_text(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        async with semaphore:
            response = await client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=kwargs.pop("temperature", 0),
                **kwargs,
            )
        return response.choices[0].message.content or ""

    return generate_text


def _is_ollama_base_url(base_url: str | None) -> bool:
    return bool(base_url and ("localhost:11434" in base_url or "127.0.0.1:11434" in base_url))


def _make_ollama_llm(*, model: str, base_url: str, max_concurrency: int):
    api_root = base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[:-3]
    chat_url = f"{api_root}/api/chat"
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    timeout_seconds = float(os.getenv("LEANRAG_OLLAMA_TIMEOUT", "600"))
    num_predict = int(os.getenv("LEANRAG_MAX_TOKENS", "2048"))

    async def generate_text(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        options = {
            "temperature": kwargs.pop("temperature", 0),
            "num_predict": int(kwargs.pop("max_tokens", num_predict)),
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }
        async with semaphore:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(chat_url, json=payload)
                response.raise_for_status()
        data = response.json()
        return (data.get("message") or {}).get("content", "")

    return generate_text


async def _extract_entities(chunk_key: str, text: str, use_llm) -> Dict[str, dict]:
    context = {
        "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        "entity_types": ",".join(PROMPTS["META_ENTITY_TYPES"]),
    }
    prompt = PROMPTS["entity_extraction"].format(**context, input_text=text)
    result = await use_llm(prompt)
    history = pack_user_ass_to_openai_messages(prompt, result)
    result += await use_llm(PROMPTS["entiti_continue_extraction"], history_messages=history)

    entities: Dict[str, dict] = {}
    for attributes in _iter_record_attributes(result, context):
        entity = await _handle_single_entity_extraction(attributes, chunk_key)
        if entity:
            entities[entity["entity_name"]] = entity
    return entities


async def _extract_relations(chunk_key: str, text: str, entities: List[str], use_llm) -> Dict[Tuple[str, str], dict]:
    if len(entities) < 2:
        return {}
    context = {
        "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        "entities": ",".join(entities),
    }
    prompt = PROMPTS["relation_extraction"].format(**context, input_text=text)
    result = await use_llm(prompt)

    relations: Dict[Tuple[str, str], dict] = {}
    for attributes in _iter_record_attributes(result, context):
        relation = await _handle_single_relationship_extraction(attributes, chunk_key)
        if relation:
            normalized = {
                "src_tgt": relation["src_id"],
                "tgt_src": relation["tgt_id"],
                "description": relation["description"],
                "weight": relation["weight"],
                "source_id": relation["source_id"],
            }
            relations[(normalized["src_tgt"], normalized["tgt_src"])] = normalized
    return relations


def _iter_record_attributes(result: str, context: dict) -> Iterable[List[str]]:
    records = split_string_by_multi_markers(
        result,
        [context["record_delimiter"], context["completion_delimiter"]],
    )
    for record in records:
        match = re.search(r"\((.*)\)", record)
        if not match:
            continue
        yield split_string_by_multi_markers(match.group(1), [context["tuple_delimiter"]])


def _dedupe_entities(results: List[Dict[str, dict]]) -> List[dict]:
    merged: Dict[str, dict] = {}
    sources = defaultdict(list)
    for result in results:
        for name, row in result.items():
            if name not in merged:
                merged[name] = dict(row)
            else:
                if row["description"] and row["description"] not in merged[name]["description"]:
                    merged[name]["description"] = f"{merged[name]['description']} | {row['description']}"[:5000]
            sources[name].append(row["source_id"])
    for name, row in merged.items():
        row["source_id"] = "|".join(dict.fromkeys(sources[name]))
        row["degree"] = 0
    return list(merged.values())


def _dedupe_relations(results: List[Dict[Tuple[str, str], dict]]) -> List[dict]:
    merged: Dict[Tuple[str, str], dict] = {}
    sources = defaultdict(list)
    for result in results:
        for key, row in result.items():
            if key not in merged:
                merged[key] = dict(row)
            else:
                merged[key]["weight"] = max(float(merged[key]["weight"]), float(row["weight"]))
                if row["description"] and row["description"] not in merged[key]["description"]:
                    merged[key]["description"] = f"{merged[key]['description']} | {row['description']}"[:5000]
            sources[key].append(row["source_id"])
    for key, row in merged.items():
        row["source_id"] = "|".join(dict.fromkeys(sources[key]))
    return list(merged.values())


def _write_jsonl(rows: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract entity/relation triples from a KG chunk file.")
    parser.add_argument("--chunks-file", required=True)
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose-logging", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose_logging else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    stats = extract_triples_for_kg(
        args.chunks_file,
        args.working_dir,
        model=args.model,
        base_url=args.base_url,
        max_concurrency=args.max_concurrency,
        limit=args.limit,
        show_progress=not args.no_progress,
        verbose_logging=args.verbose_logging,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
