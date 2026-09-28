# Datasets: what to collect and how to format it

**Short version:** the neural controller does **not** need data to train. It trains on
the validated physics model. Data is what makes the work *credible*:

1. It **validates** the model: predicted vs measured flutter speed, frequency and damping.
2. It **identifies** the real wing's parameters (this is where a genuine PINN / physics-informed
   system-identification component fits the paper).
3. It lets you **test the controllers** on the real system's behaviour.

Drop each dataset in its own folder here and run:

```
python -m flutter_ai check-data data/<your_dataset>
```

The checker validates the files, builds the wing model from your rig parameters, and
prints **predicted vs measured flutter speed**. See `data/template_synthetic/` for a
complete example; it is synthetic and labelled as such.

---

## What to look for (in priority order)

| Priority | Dataset type | Why | Where to look |
|---|---|---|---|
| ★★★ | **Wind-tunnel typical section with a trailing-edge flap** (pitch + plunge + flap), with rig parameters **and** a measured flutter speed | Exactly matches our 3-DOF model. One measured flutter point already validates (or falsifies) the model | Texas A&M "NATA" (Nonlinear Aeroelastic Test Apparatus, Strganac group; papers and theses on active flutter suppression with a TE flap). NASA Langley "BACT" (Benchmark Active Controls Technology; pitch-and-plunge apparatus with TE flap; NASA technical memoranda on the NASA Technical Reports Server). Duke University typical section with control-surface freeplay (Dowell group, late 1990s) |
| ★★★ | **Time histories** from such a rig: h(t), α(t), β(t) at several airspeeds, ideally including sub-critical decays after a tap or gust, and one run near flutter | Lets us identify damping and stiffness, check time-domain accuracy, and train/validate a physics-informed identifier | Same sources. Ask authors directly; many share data for research |
| ★★ | **V-g data**: damping ratio and frequency of each mode vs airspeed (sub-critical tests) | Validates the whole flutter curve, not just one point | Often tabulated or plotted in the same papers; digitised plots are acceptable (say so in `source`) |
| ★★ | Nonlinear stiffness (freeplay, hardening spring) and **limit-cycle oscillation** amplitude vs airspeed | Next model upgrade (Phase 7.2); where neural control should beat linear control | Duke freeplay experiments; NATA nonlinear pitch spring |
| ★ | 3D wing flutter benchmarks (AGARD 445.6 wing; Goland wing) | For the next upgrade beyond a typical section (N-mode flexible wing) | AGARD reports; widely reproduced in the CFD/aeroelastic literature |
| ★ | Your own university wind-tunnel rig | Best of all for a paper: your own experiment | Talk to your lab / advisor |

Tips:
- **One good, well-documented rig beats five partial ones.** Rig parameters without a measured
  flutter point are much less useful, and time histories without rig parameters are close to useless.
- Record the **source** (DOI or URL) and the **licence or permission**. We must be able to cite
  it and say whether it can be redistributed.
- If you digitise values from plots, say so. It limits the accuracy you can claim.

---

## Folder format

```
data/<dataset_name>/
├── metadata.json            required: rig parameters, source, licence, measured flutter point
└── runs/                    optional but valuable
    ├── <run_id>.csv         one time history per run
    └── <run_id>.json        optional notes for the run
```

### `metadata.json`

All values are **SI, per unit span**, with **angles in radians**. Sign conventions: h positive down,
α positive nose-up, β positive trailing-edge down; `a` and `flap_hinge` in semichords aft of mid-chord.

```json
{
  "name": "nata_2004_linear",
  "description": "What the rig is and which configuration",
  "source": "Author, Title, Journal/Report, Year, DOI or URL",
  "licence": "e.g. 'research use with permission of <author>, 2026-10-01'",
  "rig": {
    "b": 0.135,            "a": -0.6,          "span": 0.6,        "rho": 1.225,
    "mass": 0.0,           "S_alpha": 0.0,     "I_alpha": 0.0,
    "k_h": 0.0,            "k_alpha": 0.0,     "c_h": 0.0,         "c_alpha": 0.0,
    "flap_hinge": 0.6,     "S_beta": 0.0,      "I_beta": 0.0,      "k_beta": 0.0,   "c_beta": 0.0,
    "beta_cmd_max_deg": 15
  },
  "stiffness_nonlinearity": "e.g. 'k_alpha(alpha) = k0 + k1*alpha + k2*alpha^2 ...' or freeplay in deg, else null",
  "sensors": [{"channel": "h", "type": "LVDT", "sample_rate_hz": 1000, "noise_std": 1e-5}],
  "measured": {"flutter_speed_mps": 0.0, "flutter_frequency_hz": 0.0, "notes": "how it was determined"}
}
```

(The numbers above are placeholders, not real rig values.)

Notes:
- **Per unit span:** if the paper gives the total mass or stiffness of a rigid wing of span `s` on
  springs, divide by `s`. Say what you did in `description`.
- `S_alpha` = static moment about the elastic axis (kg·m/m), positive if the CG is **aft** of the axis.
  You may give `x_alpha` (semichords) and `r_alpha_sq` instead of `S_alpha`/`I_alpha`.
- Damping `c_*` may be given as damping ratios instead; ask and we will extend the loader.
- For a 2-DOF rig, omit all `*_beta` fields and `flap_hinge`.

### `runs/<run_id>.csv`

A header row, then one row per sample. Lines starting with `#` are comments.

| Column | Unit | Required |
|---|---|---|
| `t` | s | yes (strictly increasing; ≥ 500 Hz recommended) |
| `U` | m/s | yes (airspeed) |
| `h` | m | yes |
| `alpha` | rad | yes |
| `beta` | rad | if the rig has a flap |
| `beta_cmd` | rad | if the flap was actuated |
| `h_dot`, `alpha_dot`, `beta_dot` | m/s, rad/s | optional (we can differentiate) |
| `F_h`, `M_alpha` | N/m, N·m/m | optional (other actuators) |
| `w_gust` | m/s | optional (gust generator signal) |

### `runs/<run_id>.json` (optional)

```json
{"controller": "none | lqr | ...", "airspeed_mps": 12.3, "excitation": "tap on plunge", "notes": "..."}
```

---

## What happens with your data (plan)

1. `check-data` → model vs measurement report (flutter speed error %).
2. **Parameter identification** (Phase 4): fit uncertain parameters (damping, stiffnesses, `a`) to
   the time histories, with physics-informed regularisation. This is the legitimate "PINN"
   contribution, and the paper can report identified vs nominal values.
3. Re-run the controller benchmark on the **identified** model (`python -m flutter_ai train` +
   `evaluate` with the new preset). Results then describe the real rig, not just a textbook one.
