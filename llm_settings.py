import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _env_or_default(name: str, default: Any) -> Any:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def load_llm_settings() -> Dict[str, Any]:
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    llm_cfg = cfg.get("llm") or cfg.get("openai") or {}
    return {
        "chat_model": _env_or_default("OPENAI_CHAT_MODEL", llm_cfg.get("chat_model", "qwen2.5")),
        "extraction_model": _env_or_default("LEANRAG_EXTRACTION_MODEL", llm_cfg.get("extraction_model", "qwen3_14b")),
        "commonkg_model": _env_or_default("LEANRAG_COMMONKG_MODEL", llm_cfg.get("commonkg_model", "qwen3_32b")),
        "evaluation_model": _env_or_default("LEANRAG_EVALUATION_MODEL", llm_cfg.get("evaluation_model", "deepseek-v3-250324")),
        "embedding_model": _env_or_default("OPENAI_EMBEDDING_MODEL", llm_cfg.get("embedding_model", "bge_m3")),
        "base_url": _env_or_default("OPENAI_BASE_URL", llm_cfg.get("base_url", "http://localhost:8001/v1")),
        "api_key": _env_or_default("OPENAI_API_KEY", llm_cfg.get("api_key", "EMPTY")),
        "embedding_base_url": _env_or_default(
            "OPENAI_EMBEDDING_BASE_URL",
            llm_cfg.get("embedding_base_url", llm_cfg.get("base_url", "http://localhost:8001/v1")),
        ),
        "embedding_api_key": _env_or_default("OPENAI_EMBEDDING_API_KEY", llm_cfg.get("embedding_api_key", llm_cfg.get("api_key", "EMPTY"))),
    }
