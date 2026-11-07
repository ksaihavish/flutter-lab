# Submission checklist

*Prepared 2026-09-29. Tick items as they are done. The draft is `paper-draft.md` (generated: re-run
`python -m flutter_ai paper` after any new results).*

## Only the owner can do these

- [ ] **Advisor or mentor review** of the story and the draft (Phases.md 4.8 "done when").
- [ ] **Choose the venue** (e.g. an AIAA SciTech Forum session on aeroelasticity or controls, a controls conference,
      or a journal) and check its page limit, template and deadline on the venue's own site.
- [ ] **Author list and affiliations**; decide how AI assistance is acknowledged, following the venue's policy.
- [x] Full texts checked for every reference with an open copy (2026-09-29; `references.bib` → `readstatus`,
      metadata confirmed against Crossref / arXiv). This corrected the draft: Papacharalampopoulos et al. 2026 *do*
      let the LLM choose the mode, so the architecture pattern is theirs (see related-work.md §1).
- [ ] **Read through a library** the four paywalled papers still marked `abstract` (Bernelli-Zazzera 2000,
      Mattaboni 2009, Tang 2018, Tang 2021); the draft cites them only at title/abstract level until then.
- [ ] Read Papacharalampopoulos et al. 2026 yourself (open access): it is the paper a reviewer will compare you with.
- [ ] Confirm the Hodges & Pierce edition used for the textbook benchmark (the draft does not cite it yet).
- [ ] Decide the code and data availability statement (the repository is public; licence in `LICENSE`).

## Writing still to do (after the fault-tolerant results)

- [x] **Decide the framing and title** (owner, 2026-11-07: Guardian-first). Title "Fast models decide, the language
      model explains: ..." (`paper.TITLE`); abstract opening, research questions and contributions rewritten around the
      guardian, with the LLM-as-decision-maker results as the motivating negative result.
- [ ] **Trim the abstract** to the venue's word limit (it is ~370 words; most venues allow 150-250).
- [ ] Cite fault detection and isolation and flutter-margin monitoring work in §2 (placeholder marked *[Owner: ...]*;
      `related-work.md` §7) before claiming anything about the guardian beyond its evaluation.

- [ ] §8 Discussion and §10 Conclusion are generated from the results (data-driven bullets): edit the wording and
      write the closing sentence of §10 (marked *[Owner: ...]*).
- [ ] Abstract: re-read once §7.5 is filled (it updates automatically).
- [x] Readable PDF for advisor review: `python -m flutter_ai paper --pdf` → `paper-draft.pdf` (local build, gitignored).
- [ ] Convert to the venue template (the Markdown draft maps directly to LaTeX sections; `references.bib` is BibTeX).

## Evidence already in place

- [x] Validated model: textbook benchmark, independent p-k cross-check, thin-airfoil flap loads, NASA BACT flutter point.
- [x] Realistic actuation (servo dynamics, slew-rate limit from a datasheet) and loop delay.
- [x] Fair baselines: one shared cost; LQG; classical knobs tuned on disjoint seeds.
- [x] 20 seeds, paired Wilcoxon tests and bootstrap CIs; robustness study; ablations.
- [x] LLM effort-mode benchmark (negative result) with measured latency.
- [x] Fault-tolerant study built; dwell-time stability bound computed and enforced; tests check the analysis
      against the simulator.
- [x] Fault-tolerant benchmark results (`ft-benchmark`: 7 fault cases × 20 seeds; Memory.md §4).
- [x] Pre-registered physics-informed test on fresh seeds 100-119 (§7.6 of the draft).
- [x] Guardian (learned diagnosis + physics, no LLM): designed on seeds 1000-1009, final test once on seeds 300-319
      (§7.7).
- [x] LLM off the control path: explainer faithfulness and the post-flight reviewer with a verification gate (§7.8).

## Nice to have (strengthens the paper, not required)

- [ ] Run the fault-tolerant study with a second LLM (the Claude backend, or another local model) to show the
      result is not specific to one model.
- [ ] Measured time histories from a real rig (then `identify` → parameters with uncertainty).
- [ ] BACT compressibility correction and a second flutter point.
- [ ] Qualitative analysis of the LLM's reasoning in the decision logs (RQ4).
