# Changelog

## Command line data generation (2026-11)

- New `generate_training_data.py` and `ssrf_realtime/datagen.py`: parallel, resumable
  generation of (spectrum → Q-optimizing RF program) datasets for machine learning.
  - Two modes: random (P range, random histories) and sweep (P grid × burn histories ×
    evolution times).
  - Options for DNP on/off, RF power and energy caps, and post-pulse evolution.
  - `info` and `export` subcommands. See `docs/COMMAND_LINE_DATA_GENERATION.md`.
- Fix: the optimizer's validation replay no longer stalls when the state was captured at a
  large simulation time with a very small playback step.
- 457 tests.

## Production release (2026-11)

This replaces the earlier ssRF beta: the display/playback update with demo materials only.

### Physics
- **Spectral RF model.** Weak-drive saturation from a homogeneous Lorentzian kernel (strength κ,
  width Γ). Rates are per spin, any number of carriers add in power, and the RF is applied with
  FFT. The two overlapping transitions at one frequency belong to different spins.
- **Measured line shape.** Two-site C–D/O–D powder mixture (azimuth-averaged, with η) fitted to
  UVA d-butanol spectra. 500-bin grid at the acquisition spacing.
- **Spin diffusion** as conservative pair currents: same-branch, cross-branch and
  double-quantum channels, all conserving mass and P, and stationary at Boltzmann.
- **Per-packet mechanism profiles.** RF coupling, RF homogeneous width, exchange, DNP and T1,
  each with density, orientation (P₂/P₄ in θ) and O–D site factors. The early density-weighted
  RF law is a special case.
- **Extended engine.** Optional separate C–D/O–D populations and an optional nuclear dipolar
  reservoir (Provotorov thermal mixing). Exchange conserves energy exactly.
- **DNP** rate and saturation fitted to the 2014 build-up. Spin diffusion and RF coupling
  fitted to a 0.1 W burn/refill, with total U = 1 ≙ 0.1 W.

### Optimizer
- Exact adjoint for every mechanism, including the extended engine.
- Total-power and exposure limits; power coordinates scaled to the available power.
- Duration refinement on both sides of the best point; best iterate kept on timeout.
- Optional power-preserving smoothing of RF profiles, checked against the population model.
- Stricter endpoint validation by step halving, up to 12 halvings in the shipped materials.
- RF uses its validated fine step only while it is on; the simulation clock is unaffected.

### Tools and usability
- `run_app.py <material>` opens any shipped material by name; `--list` lists them.
- `tools/steady_state_rf_dnp.py`: best constant RF under continuous DNP, solved as an exact
  steady state with an implicit-function gradient. Writes a GUI-loadable program.
- Six materials: `d_butanol_matched` (default), `d_butanol_basic`, `d_butanol_extend`,
  `d_butanol_full_physics`, `d_butanol_same_branch_transport`, `demo_single_pake`.
- Typed numeric fields throughout; scrollable settings.
- Material and population-snapshot files are versioned and fingerprinted. Older files migrate
  with neutral defaults.

### Documentation
- New README, user guide, materials and parameter reference, physics summary, and the key
  results of the tuning studies (`docs/`). Derivations are in `docs/reference/`.

### Tests
- 453 tests, covering conservation, adjoint versus finite differences, live versus optimizer
  dynamics, Provotorov sign and energy conservation, snapshot migration, GUI playback and the
  shipped materials.
