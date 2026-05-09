import json
from typing import Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

from agents.llm_utils import parse_json_object


def schema_response_format(model: Type[BaseModel], name: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": model.model_json_schema(),
        },
    }


def generate_pydantic(
    query_service,
    *,
    model: Type[BaseModel],
    schema_name: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int = 1024,
) -> Tuple[Optional[BaseModel], Optional[str], str]:
    if not getattr(query_service, "llm_available", False):
        return None, "llm_unavailable", ""
    try:
        raw = query_service.use_llm_func(
            prompt,
            system_prompt=system_prompt,
            temperature=0,
            max_tokens=max_tokens,
            response_format=schema_response_format(model, schema_name),
        )
    except Exception as exc:
        return None, f"llm_error:{exc}", ""
    parsed, error = validate_pydantic(raw, model)
    return parsed, error, str(raw or "")


def repair_pydantic(
    query_service,
    *,
    model: Type[BaseModel],
    schema_name: str,
    raw: str,
    original_prompt: str,
    system_prompt: str = "Return only valid JSON that matches the provided schema. No markdown.",
    max_tokens: int = 1024,
) -> Tuple[Optional[BaseModel], Optional[str]]:
    repair_prompt = "\n".join(
        [
            "Convert the previous response into exactly one valid JSON object matching this JSON schema.",
            json.dumps(model.model_json_schema(), ensure_ascii=False),
            "",
            "Original task:",
            original_prompt[:2500],
            "",
            "Previous response:",
            str(raw)[:2500],
        ]
    )
    try:
        repaired = query_service.use_llm_func(
            repair_prompt,
            system_prompt=system_prompt,
            temperature=0,
            max_tokens=max_tokens,
            response_format=schema_response_format(model, f"{schema_name}-repair"),
        )
    except Exception as exc:
        return None, f"llm_repair_error:{exc}"
    return validate_pydantic(repaired, model)


def validate_pydantic(raw: str, model: Type[BaseModel]) -> Tuple[Optional[BaseModel], Optional[str]]:
    text = str(raw or "").strip()
    try:
        return model.model_validate_json(text), None
    except ValidationError as exc:
        parsed = parse_json_object(text)
        if parsed is None:
            return None, "llm_invalid_json"
        try:
            return model.model_validate(parsed), None
        except ValidationError as nested:
            return None, _validation_error(nested)
    except ValueError:
        parsed = parse_json_object(text)
        if parsed is None:
            return None, "llm_invalid_json"
        try:
            return model.model_validate(parsed), None
        except ValidationError as exc:
            return None, _validation_error(exc)


def _validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "llm_schema_validation_error"
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", [])) or "root"
    return f"llm_schema_validation_error:{loc}:{first.get('type', 'invalid')}"
