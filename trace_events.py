import json
import os
import sys
import time
import uuid
from contextvars import ContextVar
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


load_dotenv()
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - yaml is a project dependency
    yaml = None

_CURRENT_RECORDER: ContextVar[Optional["TraceRecorder"]] = ContextVar("triage_trace_recorder", default=None)
_TRACE_CONFIG: Optional[Dict[str, Any]] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _trace_config() -> Dict[str, Any]:
    global _TRACE_CONFIG
    if _TRACE_CONFIG is not None:
        return _TRACE_CONFIG
    if yaml is None:
        _TRACE_CONFIG = {}
        return _TRACE_CONFIG
    try:
        data = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
    except OSError:
        data = {}
    _TRACE_CONFIG = data.get("trace", {}) or {}
    return _TRACE_CONFIG


def current_recorder() -> Optional["TraceRecorder"]:
    return _CURRENT_RECORDER.get()


def compact_value(value: Any, max_chars: int = 1200, max_items: int = 8) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"
    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                compacted["_truncated_keys"] = len(value) - max_items
                break
            compacted[str(key)] = compact_value(item, max_chars=max_chars, max_items=max_items)
        return compacted
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        compacted = [
            compact_value(item, max_chars=max_chars, max_items=max_items)
            for item in values[:max_items]
        ]
        if len(values) > max_items:
            compacted.append({"_truncated_items": len(values) - max_items})
        return compacted
    return compact_value(str(value), max_chars=max_chars, max_items=max_items)


def _kg_ids_from_evidence(evidence: Any) -> list:
    if not isinstance(evidence, list):
        return []
    ids = []
    for row in evidence:
        if isinstance(row, dict) and row.get("kg_id") and row["kg_id"] not in ids:
            ids.append(row["kg_id"])
    return ids


def _retrieval_modes_from_evidence(evidence: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(evidence, list):
        return counts
    for row in evidence:
        if not isinstance(row, dict):
            continue
        mode = str(row.get("retrieval_mode") or "unknown")
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def summarize_kg_selection(selected: Dict[str, Any]) -> Dict[str, Any]:
    probe = selected.get("community_probe", {}) if isinstance(selected, dict) else {}
    return {
        "method": selected.get("method"),
        "primary_kg": selected.get("primary_kg"),
        "domain_kgs": selected.get("domain_kgs", []),
        "compliance_kgs": selected.get("compliance_kgs", []),
        "routing_kgs": selected.get("routing_kgs", []),
        "domain_kg_scores": compact_value(selected.get("domain_kg_scores", []), max_items=6),
        "community_probe": {
            "available": probe.get("available"),
            "method": probe.get("method"),
            "kg_scores": compact_value(probe.get("kg_scores", []), max_items=6),
            "bridge_count": len(probe.get("bridge_trajectory", []) or []),
            "hit_count": len(probe.get("community_hits", []) or []),
        },
    }


def summarize_agent_result(agent: str, result: Dict[str, Any]) -> Dict[str, Any]:
    evidence = result.get("evidence", [])
    summary: Dict[str, Any] = {
        "agent": agent,
        "confidence": result.get("confidence"),
        "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
        "evidence_kgs": _kg_ids_from_evidence(evidence),
        "retrieval_modes": _retrieval_modes_from_evidence(evidence),
    }
    if agent == "domain":
        classification = result.get("classification", {})
        summary.update(
            {
                "source": classification.get("source"),
                "product": classification.get("product"),
                "cfpb_product": classification.get("cfpb_product"),
                "issue": classification.get("issue"),
                "cfpb_issue": classification.get("cfpb_issue"),
                "cfpb_sub_issue": classification.get("cfpb_sub_issue"),
                "error": result.get("classification_error"),
            }
        )
    elif agent == "compliance":
        summary.update(
            {
                "source": result.get("source"),
                "compliance_risk": result.get("compliance_risk"),
                "veto": result.get("veto"),
                "policy_checks": result.get("policy_checks", []),
                "error": result.get("assessment_error"),
            }
        )
    elif agent == "routing":
        summary.update(
            {
                "source": result.get("source"),
                "route": result.get("route"),
                "error": result.get("route_error"),
            }
        )
    elif agent == "resolution":
        plan = result.get("resolution_plan", {})
        summary.update(
            {
                "source": result.get("source"),
                "owner_team": plan.get("owner_team"),
                "actions_count": len(plan.get("actions", []) or []),
                "customer_response": plan.get("customer_response"),
                "error": result.get("plan_error"),
            }
        )
    return compact_value(summary)


def summarize_final_result(result: Dict[str, Any]) -> Dict[str, Any]:
    trace = result.get("decision_trace", {})
    return compact_value(
        {
            "classification": result.get("classification"),
            "severity": result.get("severity"),
            "compliance_risk": result.get("compliance_risk"),
            "route": result.get("route"),
            "escalate": result.get("escalate"),
            "confidence": trace.get("confidence"),
            "uncertainty": trace.get("uncertainty"),
            "escalation_reason": trace.get("escalation_reason"),
            "policy_checks": trace.get("policy_checks", []),
        }
    )


class TraceObservation:
    def __init__(self, langfuse_observation: Any = None):
        self.langfuse_observation = langfuse_observation
        self.output: Any = None
        self.metadata: Dict[str, Any] = {}

    def update(
        self,
        output: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if output is not None:
            self.output = output
        if metadata:
            self.metadata.update(metadata)
        if self.langfuse_observation is not None:
            try:
                self.langfuse_observation.update(
                    output=compact_value(output, max_chars=3000) if output is not None else None,
                    metadata=compact_value(metadata or {}, max_chars=3000, max_items=20),
                    **kwargs,
                )
            except Exception:
                pass


class TraceRecorder:
    def __init__(
        self,
        events_path: Optional[str] = None,
        stream_events: bool = False,
        langfuse_enabled: Optional[bool] = None,
        trace_id_seed: Optional[str] = None,
        trace_name: str = "complaint-orchestrator",
        metadata: Optional[Dict[str, Any]] = None,
        capture_llm_io: Optional[bool] = None,
        llm_io_max_chars: Optional[int] = None,
    ):
        self.run_id = uuid.uuid4().hex
        self.events_path = Path(events_path) if events_path else None
        self.stream_events = stream_events
        self.trace_name = trace_name
        self.metadata = metadata or {}
        self.sequence = 0
        self.events = []
        self.started_at = time.perf_counter()
        self._depth = 0
        self._event_file = None
        self._langfuse = None
        self._langfuse_trace_id = None
        self._langfuse_enabled = _env_bool("TRIAGE_LANGFUSE", False) if langfuse_enabled is None else langfuse_enabled
        config = _trace_config()
        configured_capture = bool(config.get("capture_llm_io", False))
        configured_max_chars = int(config.get("llm_io_max_chars", 12000) or 12000)
        self.capture_llm_io = (
            _env_bool("TRIAGE_TRACE_LLM_IO", configured_capture)
            if capture_llm_io is None
            else capture_llm_io
        )
        self.llm_io_max_chars = (
            llm_io_max_chars
            if llm_io_max_chars is not None
            else _env_int("TRIAGE_TRACE_LLM_MAX_CHARS", configured_max_chars)
        )

        if self.events_path:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            self._event_file = self.events_path.open("w", encoding="utf-8")
        if self._langfuse_enabled:
            self._init_langfuse(trace_id_seed or self.run_id)

    @property
    def enabled(self) -> bool:
        return bool(self._event_file or self.stream_events or self._langfuse)

    def _init_langfuse(self, trace_id_seed: str) -> None:
        if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
            os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL", "")
        try:
            from langfuse import get_client
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Langfuse tracing was requested but the langfuse package is not installed. "
                "Install it with `pip install langfuse` or run without --langfuse."
            ) from exc
        self._langfuse = get_client()
        try:
            self._langfuse_trace_id = self._langfuse.create_trace_id(seed=trace_id_seed)
        except Exception:
            self._langfuse_trace_id = None

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None, level: str = "INFO") -> None:
        if not self.enabled:
            return
        self.sequence += 1
        event = {
            "sequence": self.sequence,
            "time": _utc_now(),
            "elapsed_ms": round((time.perf_counter() - self.started_at) * 1000, 3),
            "run_id": self.run_id,
            "trace_name": self.trace_name,
            "level": level,
            "event_type": event_type,
            "payload": compact_value(payload or {}, max_chars=self.llm_io_max_chars, max_items=20),
        }
        self.events.append(event)
        line = json.dumps(event, ensure_ascii=False)
        if self._event_file:
            self._event_file.write(line + "\n")
            self._event_file.flush()
        if self.stream_events:
            print(line, file=sys.stderr, flush=True)

    def _langfuse_context(
        self,
        name: str,
        input_value: Any,
        metadata: Dict[str, Any],
        as_type: str,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ):
        if self._langfuse is None:
            return nullcontext(None)
        kwargs: Dict[str, Any] = {
            "as_type": as_type,
            "name": name,
            "input": compact_value(input_value, max_chars=self.llm_io_max_chars),
        }
        if model:
            kwargs["model"] = model
        if model_parameters:
            kwargs["model_parameters"] = compact_value(model_parameters, max_chars=1000, max_items=20)
        merged_metadata = {**self.metadata, **metadata, "local_run_id": self.run_id}
        if merged_metadata:
            kwargs["metadata"] = compact_value(merged_metadata, max_chars=3000, max_items=20)
        if self._depth == 0 and self._langfuse_trace_id:
            kwargs["trace_context"] = {"trace_id": self._langfuse_trace_id}
        return self._langfuse.start_as_current_observation(**kwargs)

    @contextmanager
    def span(
        self,
        name: str,
        input_value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        as_type: str = "span",
    ):
        metadata = metadata or {}
        span_id = uuid.uuid4().hex[:12]
        self.emit(f"{name}.started", {"span_id": span_id, "input": input_value, "metadata": metadata})
        started = time.perf_counter()
        cm = self._langfuse_context(name, input_value, metadata, as_type)
        self._depth += 1
        token = _CURRENT_RECORDER.set(self)
        try:
            langfuse_observation = cm.__enter__()
            observation = TraceObservation(langfuse_observation)
            yield observation
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            if "langfuse_observation" in locals() and langfuse_observation is not None:
                try:
                    langfuse_observation.update(level="ERROR", status_message=str(exc)[:500])
                except Exception:
                    pass
            self.emit(
                f"{name}.failed",
                {"span_id": span_id, "duration_ms": duration_ms, "error": str(exc)},
                level="ERROR",
            )
            raise
        else:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            self.emit(
                f"{name}.completed",
                {
                    "span_id": span_id,
                    "duration_ms": duration_ms,
                    "output": observation.output,
                    "metadata": observation.metadata,
                },
            )
        finally:
            try:
                cm.__exit__(*sys.exc_info())
            finally:
                _CURRENT_RECORDER.reset(token)
                self._depth -= 1

    @contextmanager
    def generation(
        self,
        name: str,
        input_value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ):
        metadata = metadata or {}
        generation_id = uuid.uuid4().hex[:12]
        self.emit(
            f"{name}.started",
            {
                "generation_id": generation_id,
                "input": input_value,
                "metadata": metadata,
                "model": model,
                "model_parameters": model_parameters,
            },
        )
        started = time.perf_counter()
        cm = self._langfuse_context(
            name,
            input_value,
            metadata,
            "generation",
            model=model,
            model_parameters=model_parameters,
        )
        self._depth += 1
        token = _CURRENT_RECORDER.set(self)
        try:
            langfuse_observation = cm.__enter__()
            observation = TraceObservation(langfuse_observation)
            yield observation
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            if "langfuse_observation" in locals() and langfuse_observation is not None:
                try:
                    langfuse_observation.update(level="ERROR", status_message=str(exc)[:500])
                except Exception:
                    pass
            self.emit(
                f"{name}.failed",
                {"generation_id": generation_id, "duration_ms": duration_ms, "error": str(exc)},
                level="ERROR",
            )
            raise
        else:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            self.emit(
                f"{name}.completed",
                {
                    "generation_id": generation_id,
                    "duration_ms": duration_ms,
                    "output": observation.output,
                    "metadata": observation.metadata,
                    "model": model,
                },
            )
        finally:
            try:
                cm.__exit__(*sys.exc_info())
            finally:
                _CURRENT_RECORDER.reset(token)
                self._depth -= 1

    def llm_input(
        self,
        prompt: Any,
        system_prompt: Optional[str] = None,
        history_messages: Optional[Any] = None,
    ) -> Dict[str, Any]:
        prompt_text = str(prompt or "")
        system_text = str(system_prompt or "")
        history_count = len(history_messages or []) if isinstance(history_messages, list) else 0
        if self.capture_llm_io:
            return {
                "system_prompt": self.truncate_text(system_text),
                "prompt": self.truncate_text(prompt_text),
                "history_messages": compact_value(history_messages or [], max_chars=self.llm_io_max_chars),
            }
        return {
            "system_prompt_chars": len(system_text),
            "prompt_chars": len(prompt_text),
            "history_messages_count": history_count,
            "input_capture": "set TRIAGE_TRACE_LLM_IO=true or pass --trace-llm-io to capture prompt bodies",
        }

    def llm_output(self, output: Any) -> Any:
        return self.truncate_text(str(output or ""))

    def truncate_text(self, text: str) -> str:
        if len(text) <= self.llm_io_max_chars:
            return text
        return f"{text[:self.llm_io_max_chars]}... [truncated {len(text) - self.llm_io_max_chars} chars]"

    def close(self) -> None:
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception:
                pass
        if self._event_file:
            self._event_file.close()
            self._event_file = None
