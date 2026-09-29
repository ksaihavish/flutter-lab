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

from .modes import MODES, Decision, make_decision
from .report import SituationFeatures, format_report
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
    """Backend-independent safety wrapper. Subclasses implement _complete(report) -> JSON text."""

    name = "llm"
    backend = ""
    transport_errors: tuple[type[BaseException], ...] = ()

    def __init__(self, model: str | None = None, timeout_s: float | None = None, max_calls: int = 40):
        self.model = model or llm_model(self.backend)
        self.timeout_s = timeout_s or float(os.environ.get("FLUTTER_AI_LLM_TIMEOUT", "30"))
        self.max_calls = max_calls  # cost / run-time guard per simulation run
        self.fallback = RuleSupervisor()
        self.calls = 0
        self.failures = 0

    def reset(self) -> None:
        self.fallback.reset()
        self.calls = 0
        self.failures = 0

    def _complete(self, report: str) -> str:
        raise NotImplementedError

    def decide(self, f: SituationFeatures) -> Decision:
        report = format_report(f)
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
            if mode not in MODES:
                raise ValueError(f"invalid mode {mode!r}")
            return make_decision(f.time, mode, reasoning, source="llm", latency_s=latency)
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

    def _body(self, report: str, **options) -> dict:
        return {
            "model": self.model,
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "keep_alive": "10m",  # stay loaded between decisions (a cold load takes ~15 s)
            "options": {"temperature": 0, "num_ctx": 2048, **options},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": report}],
        }

    def _complete(self, report: str) -> str:
        return self._transport(self._body(report))["message"]["content"]


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

    def _complete(self, report: str) -> str:
        response = self._get_client().beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": report}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
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
