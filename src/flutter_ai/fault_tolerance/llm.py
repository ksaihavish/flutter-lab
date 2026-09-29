"""
The LLM supervisor for fault-tolerant supervision: the same safety wrapper as the effort-mode supervisor
(schema-constrained enum, validation, rule fallback, measured latency), pointed at the fault-tolerant modes,
the diagnostic report and the fault-tolerant rules.

The guidance states general engineering principles only. It does not mention the held-out faults (loop delay,
flap damage), so the LLM has to generalise from symptoms, like a pilot would.
"""

from __future__ import annotations

from ..supervisor.llm import LLMSupervisor, make_llm_supervisor
from .diagnostics import FTFeatures, format_ft_report
from .simulate import FT_MODES, MODE_ORDER, FTDecision
from .supervisors import FTRuleSupervisor

FT_SYSTEM_PROMPT = f"""You are the supervisory layer of an active flutter-suppression system on a small UAV wing.
The wing cruises above its flutter speed: without active control it would flutter and break apart. A fast
controller (200 Hz) moves a flap to keep the wing stable. Every few seconds you receive a status report and choose
the operating mode. You never command the flap directly.

Modes:
{chr(10).join(f"- {name}: {spec['description']}" for name, spec in FT_MODES.items())}

Guidance:
- Stay in 'nominal' while everything is normal: slowing down costs mission time.
- The report compares the flight computer's model with the measurements. Values near x1 of the healthy baseline
  are normal. A clear, persistent rise means something in the aircraft has changed.
- A large, persistent drop in the pitch oscillation frequency suggests structural damage, which lowers the
  flutter speed itself. Small wandering of the frequency is normal.
- If the flap cannot follow its commands, or the response grows, control authority is being lost: slow down.
- Prefer the smallest change that removes the risk; escalate if the risk keeps growing.

Respond with the mode and one or two sentences of reasoning that cite the report."""


def make_ft_llm_supervisor(backend: str | None = None, **kwargs) -> LLMSupervisor:
    sup = make_llm_supervisor(backend, **kwargs)
    sup.configure(system_prompt=FT_SYSTEM_PROMPT, modes=MODE_ORDER, format=format_ft_report,
                  make=lambda f, mode, reasoning, source, latency: FTDecision(
                      time=f.time, mode=mode, reasoning=reasoning, source=source, latency_s=latency),
                  fallback=FTRuleSupervisor())
    sup.name = "llm"
    return sup


__all__ = ["FT_SYSTEM_PROMPT", "FTFeatures", "make_ft_llm_supervisor"]
