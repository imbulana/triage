import json
from typing import Dict, List, Optional


def evidence_summary(evidence: List[Dict], max_items: int = 4) -> str:
    rows = []
    for item in evidence[:max_items]:
        kg_id = item.get("kg_id", "unknown")
        if "error" in item:
            rows.append(f"- {kg_id}: error={item['error']}")
            continue
        entities = ", ".join(item.get("entities", [])[:8])
        response = " ".join(str(item.get("response", "")).split())[:500]
        rows.append(f"- {kg_id}: entities=[{entities}] response={response}")
    return "\n".join(rows) if rows else "- no KG evidence"


def parse_json_object(raw: str) -> Optional[Dict]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def clamp_float(value, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def clean_string_list(value, fallback: Optional[List[str]] = None, limit: int = 8) -> List[str]:
    if not isinstance(value, list):
        return list(fallback or [])[:limit]
    cleaned = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if value is None:
        return default
    return bool(value)
