"""
Claude-backed supervisor.

Safety design (see docs/Rules.md):
- The LLM only chooses a mode from a fixed enum (structured JSON output); it
  never commands forces. Invalid, refused, slow or failed calls fall back to
  the rule-based supervisor, and the fallback is recorded.
- Wall-clock latency is measured and, by default, applied as a delay in
  simulation time, so results reflect what a real slow-loop supervisor would do.

Configuration (environment):
    ANTHROPIC_API_KEY        credentials (or an `ant auth login` profile)
    FLUTTER_AI_LLM_MODEL     model id, default "claude-opus-5"
    FLUTTER_AI_LLM_TIMEOUT   per-request timeout in seconds, default 30
"""

from __future__ import annotations

import json
import logging
import os
import time

from .modes import MODES, Decision, make_decision
from .report import SituationFeatures, format_report
from .rules import RuleSupervisor

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"

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


def llm_status() -> dict:
    try:
        import anthropic  # noqa: F401

        installed = True
    except ImportError:
        installed = False
    return {
        "installed": installed,
        "api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        "model": os.environ.get("FLUTTER_AI_LLM_MODEL", DEFAULT_MODEL),
    }


class LLMSupervisor:
    name = "llm"

    def __init__(self, model: str | None = None, timeout_s: float | None = None, client=None, max_calls: int = 40):
        self.model = model or os.environ.get("FLUTTER_AI_LLM_MODEL", DEFAULT_MODEL)
        self.timeout_s = timeout_s or float(os.environ.get("FLUTTER_AI_LLM_TIMEOUT", "30"))
        self.max_calls = max_calls  # cost guard per simulation run
        self.fallback = RuleSupervisor()
        self._client = client
        self.calls = 0
        self.failures = 0

    def reset(self) -> None:
        self.fallback.reset()
        self.calls = 0
        self.failures = 0

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=1)
        return self._client

    def _request(self, report: str):
        client = self._get_client()
        return client.beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": report}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def decide(self, f: SituationFeatures) -> Decision:
        try:
            import anthropic

            api_errors: tuple[type[Exception], ...] = (anthropic.APIError,)
        except ImportError:
            api_errors = ()

        report = format_report(f)
        start = time.perf_counter()
        try:
            if self.calls >= self.max_calls:
                raise ValueError(f"LLM call budget of {self.max_calls} per run reached")
            self.calls += 1
            response = self._request(report)
            latency = time.perf_counter() - start
            if response.stop_reason == "refusal":
                raise ValueError("model declined the request (stop_reason=refusal)")
            text = next(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
            mode, reasoning = data["mode"], str(data["reasoning"]).strip()
            if mode not in MODES:
                raise ValueError(f"invalid mode {mode!r}")
            return make_decision(f.time, mode, reasoning, source="llm", latency_s=latency)
        except (*api_errors, ImportError, ValueError, KeyError, StopIteration, TypeError) as exc:
            latency = time.perf_counter() - start
            self.failures += 1
            log.warning("LLM supervisor failed (%s); using rule-based fallback", exc)
            d = self.fallback.decide(f)
            d.source = "llm_fallback"
            d.reasoning = f"[LLM unavailable: {type(exc).__name__}] {d.reasoning}"
            d.latency_s = latency
            return d
