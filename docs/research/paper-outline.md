# Paper outline (revised 2026-09-29)

The draft itself is **generated**: `python -m flutter_ai paper` writes `paper-draft.md` from the result files, with
the prose in `src/flutter_ai/paper.py`. This file explains the story, what each section must show, and what is still
open. Rules: no hand-typed numbers (Rules.md §1); cite only what was opened (`references.bib`, `readstatus`).

## Working title

*On-board language-model supervision for active flutter suppression: a fair benchmark with fault tolerance and a
dwell-time guarantee.* Avoid "PINN" (the method is differentiable-simulation training) and avoid "first" unless a
full literature search confirms it (related-work.md).

## The story in five sentences

1. LLMs are proposed as supervisors of control loops, but their latency and unreliability clash with fast,
   safety-critical dynamics; flutter suppression is a sharp test case.
2. On a validated, realistic benchmark with fair baselines, a physics-trained neural policy beats plain LQR but not
   a turbulence-aware LQG, and each controller has its own regime (robustness, delay, actuator limits).
3. An on-board LLM supervising *effort modes* is useless: its seconds-long latency cannot act on sub-second gusts,
   and the knob barely moves the trade-off (a clean negative result).
4. Moved to *fault-tolerant supervision* (choosing among pre-verified modes after slow, persistent faults), the LLM
   is evaluated against no, random, rule-based and oracle supervision, including held-out faults. **Result:** LLM 27 %, rules 48 %, oracle 100 % of faulty flights survived (LLM vs rules: exact McNemar p = 0.0026).
5. The architecture (bounded modes, schema, fallback, one question at a time, controller bank, dwell time) makes the
   LLM's worst case analysable: no sequence of answers can destabilise the nominal closed loop.

## Sections and what each must show

| § | Content | Evidence (file) |
|---|---|---|
| 1 | Flutter, active suppression, learned control, LLM supervision; the gap; contributions | related-work.md |
| 2 | Related work: neural flutter control (not novel; Scott & Pado scheduled gains on the same BACT wing), LLM supervision (Papacharalampopoulos 2026 already has an LLM choosing among pre-verified controllers with dwell-time switching, for process systems: claim the plant, on-board latency and evaluation, not the pattern), switching stability | references.bib, related-work.md |
| 3 | Model, actuator realism, validation (p-k cross-check, BACT) | results.json `flutter`; validation/bact.json |
| 4 | Controllers, shared cost, fairness (tuned knobs on disjoint seeds) | classical_tuning_*.json |
| 5 | Supervisory architecture + dwell-time guarantee | ft_stability.json |
| 6 | Experimental protocol (scenarios, seeds, statistics) | results.json `metadata` |
| 7.1–7.3 | Controller comparison, robustness, ablations | results.json, robustness.json, ablations.json |
| 7.4 | LLM with effort modes (negative) | llm_benchmark.json |
| 7.5 | Fault-tolerant supervision (headline) | ft_benchmark.json |
| 8–10 | Discussion, limitations, conclusion | prose in paper.py; write §8 fault-tolerance bullet and §10 after reading 7.5 |

## Figures (copied to `figures/` by the generator)

V-g diagrams (uav_flap, BACT); a turbulence time history; the effort–response trade-off with supervised points;
delay robustness; the robustness panel; training curves; LLM latency + trade-off; fault-tolerant survival and
mission cost.

## Open items before submission

See `submission-checklist.md`.
