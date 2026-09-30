"""
LLM supervisor with two interchangeable backends behind one safety wrapper:

    ollama  a local model served by Ollama (runs on board: no internet, no per-call cost). Default.
    claude  Claude through the Anthropic API.

Safety design (see docs/Rules.md), identical for both backends:
- The LLM only chooses a mode from a fixed enum (schema-constrained JSON output); it never
  commands forces. Invalid, refused, slow or failed calls fall back to the rule-based
  supervisor, and the fallback is recorded.
- Wall-clock latency is measured and, by default, applied as a delay in simulation time, so
  results reflect what a real slow-loop supervisor would do. The simulator does not ask a new
  question while an answer is still pending (simulation.py).
- Off the control path, the same wrapper answers other schema-constrained questions (`ask`: the
  explainer and the post-flight reviewer of the fault-tolerant study). Every answer is checked
  again in Python against its schema (`check_schema`); a failed or invalid answer is returned as a
  failure with its reason, never silently accepted. The caller decides the fallback.

Configuration (environment):
    FLUTTER_AI_LLM_BACKEND   "ollama" (default) or "claude"
    FLUTTER_AI_LLM_MODEL     model; default "llama3.1:8b" (ollama) / "claude-opus-5" (claude)
    FLUTTER_AI_LLM_TIMEOUT   per-request timeout in seconds, default 30
    FLUTTER_AI_OLLAMA_URL    Ollama server, default http://127.0.0.1:11434
    ANTHROPIC_API_KEY        claude backend only (or an `ant auth login` profile)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from .modes import MODES, make_decision
from .report import format_report
from .rules import RuleSupervisor

log = logging.getLogger(__name__)

BACKENDS = ("ollama", "claude")
DEFAULT_BACKEND = "ollama"
DEFAULT_MODELS = {"ollama": "llama3.1:8b", "claude": "claude-opus-5"}
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

SYSTEM_PROMPT = f"""You are the supervisory layer of an active flutter-suppression system on a wing section.
A fast feedback controller (running at 200 Hz) keeps the wing stable. Every so often you receive a
status report and choose the controller's operating mode. You do not command forces directly.

Modes (least to most actuator effort):
{chr(10).join(f"- {name}: {spec['description']}" for name, spec in MODES.items())}

Guidance:
- Escalate promptly when energy is growing or the response approaches structural limits.
- De-escalate to save actuator effort only when the response is small and not growing.
- Frequent switching is undesirable; keep the current mode unless there is a reason to change.

Respond with the mode and one or two sentences of reasoning grounded in the report."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": list(MODES)},
        "reasoning": {"type": "string"},
    },
    "required": ["mode", "reasoning"],
    "additionalProperties": False,
}
# Local model context (tokens). Every request uses the same value: a different num_ctx makes Ollama reload the model.
OLLAMA_NUM_CTX = 2048
# Longest system + user prompt `ask` sends to the local model: ~3 characters per token for reports full of numbers,
# leaving room for the answer. Longer prompts are refused, never silently truncated by the server.
OLLAMA_MAX_PROMPT_CHARS = 5200

_JSON_TYPES: dict[str, tuple[type, ...]] = {"object": (dict,), "array": (list,), "string": (str,), "boolean": (bool,),
                                            "integer": (int,), "number": (int, float)}


def check_schema(value, schema: dict, path: str = "$") -> None:
    """
    Raise ValueError unless `value` satisfies `schema` (the subset of JSON Schema used in this project: type, enum,
    properties, required, additionalProperties, minimum/maximum, minLength/maxLength, items). The backends constrain
    decoding to the schema; this repeats the check in Python, because constrained decoding is not proof.
    """
    kind = schema.get("type")
    if kind is not None:
        ok = isinstance(value, _JSON_TYPES[kind]) and not (kind in ("integer", "number") and isinstance(value, bool))
        if not ok:
            raise ValueError(f"{path}: expected {kind}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{path}: {value!r} is not a finite number")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path}: longer than {schema['maxLength']} characters ({len(value)})")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        missing = [k for k in schema.get("required", ()) if k not in value]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        extra = [k for k in value if k not in props]
        if extra and schema.get("additionalProperties", True) is False:
            raise ValueError(f"{path}: unexpected {extra}")
        for k, v in value.items():
            if k in props:
                check_schema(v, props[k], f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            check_schema(v, schema["items"], f"{path}[{i}]")


@dataclass
class StructuredAnswer:
    """One schema-constrained answer from `LLMSupervisor.ask`. data is None exactly when the call failed."""

    data: dict | None
    error: str | None  # why it failed ("TimeoutError: ...", "ValueError: $.mode: ..."); None on success
    latency_s: float
    raw: str | None = None  # the model's text as received (kept for audit, also when it failed validation)
    meta: dict = field(default_factory=dict)  # backend details, e.g. Ollama's prompt_eval_count


def llm_backend() -> str:
    return os.environ.get("FLUTTER_AI_LLM_BACKEND", DEFAULT_BACKEND).strip().lower()


def llm_model(backend: str | None = None) -> str:
    backend = backend or llm_backend()
    return os.environ.get("FLUTTER_AI_LLM_MODEL") or DEFAULT_MODELS.get(backend, "")


def ollama_url() -> str:
    return os.environ.get("FLUTTER_AI_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def _ollama_models(url: str, timeout_s: float = 1.0) -> list[str] | None:
    """Models installed on the Ollama server, or None if it is not reachable."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout_s) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, KeyError):
        return None


def llm_status() -> dict:
    """What the web app shows: which backend, whether it can be used, and how to fix it if not."""
    backend, model = llm_backend(), llm_model()
    if backend == "claude":
        try:
            import anthropic  # noqa: F401

            installed = True
        except ImportError:
            installed = False
        key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        detail = ("" if installed and key else
                  "Install the llm extra (pip install -e .[llm])" if not installed else
                  "Set ANTHROPIC_API_KEY before starting the server")
        return {"backend": backend, "model": model, "label": f"Claude ({model})", "available": installed and key,
                "detail": detail, "installed": installed, "api_key_set": key}
    if backend == "ollama":
        models = _ollama_models(ollama_url())
        pulled = models is not None and (model in models or f"{model}:latest" in models)
        detail = ("" if pulled else
                  f"Start Ollama ({ollama_url()} is not reachable)" if models is None else
                  f"Run: ollama pull {model}")
        return {"backend": backend, "model": model, "label": f"Local LLM ({model})", "available": pulled,
                "detail": detail}
    return {"backend": backend, "model": model, "label": "LLM", "available": False,
            "detail": f"Unknown FLUTTER_AI_LLM_BACKEND '{backend}'. Use one of {BACKENDS}."}


class LLMSupervisor:
    """
    Backend-independent safety wrapper. Subclasses implement _complete(report) -> JSON text.

    The task is configurable (the fault-tolerant study uses its own modes, report and rules): `configure` sets the
    system prompt, the enum of valid modes (enforced by the JSON schema and again in Python), the report formatter,
    the decision constructor and the rule-based fallback. Defaults: the effort-mode supervisor.
    """

    name = "llm"
    backend = ""
    transport_errors: tuple[type[BaseException], ...] = ()

    def __init__(self, model: str | None = None, timeout_s: float | None = None, max_calls: int = 40):
        self.model = model or llm_model(self.backend)
        self.timeout_s = timeout_s or float(os.environ.get("FLUTTER_AI_LLM_TIMEOUT", "30"))
        self.max_calls = max_calls  # cost / run-time guard per simulation run
        self.system_prompt = SYSTEM_PROMPT
        self.schema: dict = RESPONSE_SCHEMA
        self.valid_modes: tuple[str, ...] = tuple(MODES)
        self.format: Callable = format_report
        self.make: Callable = lambda f, mode, reasoning, source, latency: make_decision(
            f.time, mode, reasoning, source=source, latency_s=latency)
        self.fallback = RuleSupervisor()
        self.calls = 0
        self.failures = 0
        self.last_meta: dict = {}

    def configure(self, *, system_prompt: str, modes, format: Callable, make: Callable, fallback) -> "LLMSupervisor":
        """Point the same safety wrapper at another task (another enum of modes, report and fallback)."""
        self.system_prompt, self.valid_modes = system_prompt, tuple(modes)
        props: dict = dict(RESPONSE_SCHEMA["properties"])  # type: ignore[call-overload]
        props["mode"] = {"type": "string", "enum": list(modes)}
        self.schema = {**RESPONSE_SCHEMA, "properties": props}
        self.format, self.make, self.fallback = format, make, fallback
        return self

    def reset(self) -> None:
        if hasattr(self.fallback, "reset"):
            self.fallback.reset()
        self.calls = 0
        self.failures = 0

    def _complete(self, report: str) -> str:
        return self._complete_with(self.system_prompt, report, self.schema)

    def _complete_with(self, system_prompt: str, user: str, schema: dict) -> str:
        raise NotImplementedError

    def ask(self, system_prompt: str, user: str, schema: dict,
            check: Callable[[dict], None] | None = None) -> StructuredAnswer:
        """
        One schema-constrained question off the control path (explainer, post-flight reviewer), with the decision
        path's guards: call budget, total-time timeout, transport errors, invalid JSON, the schema checked again in
        Python and an optional task check (raise ValueError). A failure is returned with its reason; the caller
        decides the fallback (it must never use a failed answer).
        """
        caught: tuple[type[BaseException], ...] = (*self.transport_errors, TimeoutError, ImportError, ValueError,
                                                   KeyError, StopIteration, TypeError)
        start, raw = time.perf_counter(), None
        self.last_meta = {}
        try:
            if self.calls >= self.max_calls:
                raise ValueError(f"LLM call budget of {self.max_calls} reached")
            self.calls += 1
            raw = self._complete_with(system_prompt, user, schema)
            latency = time.perf_counter() - start
            if latency > self.timeout_s:
                raise TimeoutError(f"answer took {latency:.1f} s (timeout {self.timeout_s:.0f} s)")
            data = json.loads(raw)
            check_schema(data, schema)
            if check is not None:
                check(data)
            return StructuredAnswer(data, None, latency, raw, dict(self.last_meta))
        except caught as exc:
            latency = time.perf_counter() - start
            if isinstance(exc, TimeoutError):
                latency = min(latency, self.timeout_s)
            self.failures += 1
            log.warning("LLM (%s) structured answer rejected (%s: %s)", self.backend, type(exc).__name__, exc)
            return StructuredAnswer(None, f"{type(exc).__name__}: {exc}", latency, raw, dict(self.last_meta))

    def decide(self, f):
        report = self.format(f)
        caught: tuple[type[BaseException], ...] = (*self.transport_errors, TimeoutError, ImportError, ValueError,
                                                   KeyError, StopIteration, TypeError)
        start = time.perf_counter()
        try:
            if self.calls >= self.max_calls:
                raise ValueError(f"LLM call budget of {self.max_calls} per run reached")
            self.calls += 1
            text = self._complete(report)
            latency = time.perf_counter() - start
            if latency > self.timeout_s:  # socket timeouts only bound silences, not the whole answer
                raise TimeoutError(f"answer took {latency:.1f} s (timeout {self.timeout_s:.0f} s)")
            data = json.loads(text)
            mode, reasoning = data["mode"], str(data["reasoning"]).strip()
            if mode not in self.valid_modes:
                raise ValueError(f"invalid mode {mode!r}")
            return self.make(f, mode, reasoning, "llm", latency)
        except caught as exc:
            # A real system gives up at the timeout, so a timed-out call costs exactly timeout_s.
            latency = time.perf_counter() - start
            if isinstance(exc, TimeoutError):
                latency = min(latency, self.timeout_s)
            self.failures += 1
            log.warning("LLM supervisor (%s) failed (%s); using rule-based fallback", self.backend, exc)
            d = self.fallback.decide(f)
            d.source = "llm_fallback"
            d.reasoning = f"[LLM unavailable: {type(exc).__name__}] {d.reasoning}"
            d.latency_s = latency
            return d


class OllamaSupervisor(LLMSupervisor):
    """Local model via Ollama's native REST API (standard library only; schema-constrained decoding)."""

    backend = "ollama"
    transport_errors = (urllib.error.URLError, TimeoutError, ConnectionError)

    def __init__(self, model: str | None = None, timeout_s: float | None = None, max_calls: int = 40,
                 url: str | None = None, transport: Callable[[dict], dict] | None = None):
        super().__init__(model, timeout_s, max_calls)
        self.url = (url or ollama_url()).rstrip("/")
        self._transport = transport or self._post

    def reset(self) -> None:
        super().reset()
        # Load the model before the run starts (as it would be before take-off): a cold load takes ~15-20 s and
        # would otherwise be charged to the first decision. A failure here resurfaces, with fallback, in decide().
        # Same options as a real request (a different num_ctx would reload the model) and the same system prompt
        # (so its processed prefix is cached), but only one output token.
        try:
            self._transport(self._body("Warm-up: reply with any valid mode.", num_predict=1))
        except self.transport_errors as exc:
            log.warning("Could not preload local model %s (%s)", self.model, exc)

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def _body(self, report: str, *, system: str | None = None, schema: dict | None = None, **options) -> dict:
        return {
            "model": self.model,
            "stream": False,
            "format": self.schema if schema is None else schema,
            "keep_alive": "10m",  # stay loaded between decisions (a cold load takes ~15 s)
            "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX, **options},
            "messages": [{"role": "system", "content": self.system_prompt if system is None else system},
                         {"role": "user", "content": report}],
        }

    def _complete(self, report: str) -> str:  # the validated decision path, unchanged (no prompt-length check)
        return self._transport(self._body(report))["message"]["content"]

    def _complete_with(self, system_prompt: str, user: str, schema: dict) -> str:
        n = len(system_prompt) + len(user)
        if n > OLLAMA_MAX_PROMPT_CHARS:
            raise ValueError(f"prompt of {n} characters may not fit the local model's {OLLAMA_NUM_CTX}-token context "
                             f"(limit {OLLAMA_MAX_PROMPT_CHARS}); shorten it rather than let the server truncate it")
        resp = self._transport(self._body(user, system=system_prompt, schema=schema))
        self.last_meta = {k: resp[k] for k in ("prompt_eval_count", "eval_count") if k in resp}
        return resp["message"]["content"]


class ClaudeSupervisor(LLMSupervisor):
    """Claude via the official Anthropic SDK."""

    backend = "claude"

    def __init__(self, model: str | None = None, timeout_s: float | None = None, client=None, max_calls: int = 40):
        super().__init__(model, timeout_s, max_calls)
        self._client = client
        try:
            import anthropic

            self.transport_errors = (anthropic.APIError,)
        except ImportError:
            self.transport_errors = ()

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=1)
        return self._client

    def _complete_with(self, system_prompt: str, user: str, schema: dict) -> str:
        response = self._get_client().beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise ValueError("model declined the request (stop_reason=refusal)")
        return next(b.text for b in response.content if b.type == "text")


def make_llm_supervisor(backend: str | None = None, **kwargs) -> LLMSupervisor:
    backend = (backend or llm_backend()).lower()
    if backend == "ollama":
        return OllamaSupervisor(**kwargs)
    if backend == "claude":
        return ClaudeSupervisor(**kwargs)
    raise ValueError(f"Unknown LLM backend '{backend}'. Use one of {BACKENDS} (FLUTTER_AI_LLM_BACKEND).")
