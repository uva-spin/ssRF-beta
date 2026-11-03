# ssRF — spin-1 selective-RF tensor-polarization simulator

ssRF simulates the level populations of deuterons (spin-1 nuclei) in a dynamically polarized
solid target, such as deuterated butanol or ND₃ at 1 K and 5 T. It shows how radio-frequency
(RF) manipulation changes the vector polarization **P** and the tensor polarization **Q**, and
it designs RF programs that maximize Q from a given starting state.

## What it does

The spectrum is a powder Pake doublet. Every frequency packet holds the populations of its
three levels (m = +1, 0, −1), and these evolve under:

| Process | What it does |
|---|---|
| **ss-RF saturation** | Weak-drive transition rates from any number of RF carriers, with a finite homogeneous linewidth. The I+ and I− transitions at the same frequency belong to different spins. |
| **AFP** | Instantaneous population inversion over chosen frequency windows. |
| **Spin diffusion** | Conservative flip-flop exchange: same-branch, cross-branch and double-quantum channels. |
| **DNP** | Build-up toward a spin-temperature target. |
| **T1** | Relaxation toward the lattice. |
| **Optional extras** | Separate C–D and O–D populations, and a nuclear dipolar reservoir (thermal mixing). |

You can:

- **watch a burn in real time:** spectrum, tensor spectrum (I+ − I−) and time traces, with
  P and Q read out live;
- **design the best single RF burn** from the current state, using a gradient-based optimizer
  with optional power and energy limits;
- **find the best constant RF while DNP runs** (steady operation), with
  `tools/steady_state_rf_dnp.py`;
- **adjust the physics** through material files: line shape, RF coupling and linewidth,
  diffusion channels, DNP, T1, and per-packet orientation, site and density dependence.

The deuterated-butanol materials are fitted to UVA target data:

| Data | Fit quality |
|---|---|
| Line shape | held-out RMS 0.8% |
| DNP build-up | held-out RMS 0.8% |
| 0.1 W RF burn and refill | held-out RMSE 3.4×10⁻⁴ |

## Install

Requires Python 3.9 or newer.

```bash
git clone https://github.com/uva-spin/ssRF-beta.git
cd ssRF-beta
python3 -m venv .venv && source .venv/bin/activate      # optional
python3 -m pip install -r requirements.txt              # numpy, scipy, matplotlib, PyQt6
```

`requirements-core.txt` installs only the numerical dependencies, for headless use. PySide6 or
PyQt5 can replace PyQt6.

## Quick start

```bash
python3 run_app.py                           # opens d_butanol_matched (default)
python3 run_app.py d_butanol_basic           # any material by name
python3 run_app.py --list                    # list the materials
python3 run_app.py d_butanol_matched --program programs/reference_0p1W_burn.json
```

The window opens **paused, with RF off**. A typical session:

1. **Design a burn:** **Generate tensor program from CURRENT state…** → **Calculate from
   captured state** → **Load generated program (do not run)**.
2. **Run it:** tick **Pause at program end**, press **RF ON**, then **Run simulation**. Q is
   held at the endpoint.
3. **Change the physics:** use **RF saturation / spin diffusion…** or **Edit ALL material /
   search settings…**, then **Save as…** to keep the result.

[docs/USER_GUIDE.md](docs/USER_GUIDE.md) describes every control.

## Materials

All materials start at P = 0.50, where the Boltzmann value is Q₀ = 0.197. The last column is
the tensor designer's result from that state with no power cap.

| Material | Use it for | Single-burn Q |
|---|---|---|
| `d_butanol_matched` | **Default.** Best fit to the measured line shape, DNP build-up and 0.1 W burn/refill. | 0.248 |
| `d_butanol_basic` | Same fitted physics. Per-spin RF selectivity, which the data do not constrain, set moderately toward the horn transitions (orientation-dependent T₂ plus mild density coupling). | 0.265 |
| `d_butanol_extend` | Same fitted physics with the selectivity pushed further. An upper-range setting; the reference burn still matches the reported ~80% depletion in 3 s. | 0.275 |
| `d_butanol_full_physics` | Matched fit plus separate C–D/O–D populations and a nuclear dipolar reservoir. | ≈ 0.248 |
| `d_butanol_same_branch_transport` | **Hypothesis:** spin diffusion stays within each line (see the limits below). | — |
| `demo_single_pake` | Generic single-site Pake doublet for teaching. | — |

[docs/MATERIALS.md](docs/MATERIALS.md) gives the fit details, the assumptions behind each
material and every parameter.

**Power limits.** The materials start with no total-power cap. To apply the experimental
limits in the tensor designer:

- tick **Limit total RF power** and enter 1, which is 0.1 W under the burn/refill calibration;
- tick **Limit total RF exposure** and enter 34, which is 3.4 J.

## Continuous DNP + RF (steady operation)

The GUI optimizer designs single burns. For the best **constant** RF while DNP runs:

```bash
python3 tools/steady_state_rf_dnp.py d_butanol_matched --dnp-target 0.5 --out programs/steady.json
python3 run_app.py d_butanol_matched --program programs/steady.json
```

Then turn **DNP ON**, press **RF ON** and **Run**. Q approaches its steady value over a few
DNP time constants (about 435 s each). With the DNP target at P = 0.50, DNP alone holds
Q = 0.197:

| Material | Steady Q with the best constant RF |
|---|---|
| `d_butanol_matched`, `d_butanol_basic` | 0.202 |
| `d_butanol_same_branch_transport` | 0.241 (0.277 with 10× `diffusion_scale`) |

## Command line

```bash
python3 design_tensor.py --material d_butanol_basic --out my_design   # optimize a burn headlessly
python3 design_tensor.py --replay my_design                           # replay and check it
python3 tools/plot_tensor_design.py my_design                         # plot the design
```

**[Command line data generation](docs/COMMAND_LINE_DATA_GENERATION.md):** generate datasets
of spectra and their Q-optimizing RF programs for training machine-learning models
(`generate_training_data.py`).

**[Training a pulse-design model](docs/TRAINING_PULSE_MODEL.md):** train a neural network that
reads a spectrum and outputs the Q-optimizing RF program.

**[Training a polarization model](docs/TRAINING_POLARIZATION_MODEL.md):** train a neural network
that reads a spectrum, manipulated or not, and outputs the vector and tensor polarization.

**[Notes](notes/):** technical notes on the physics and implementation of the simulator and on
spin-1 rate equations (LaTeX and PDF).

## Layout

| Path | Contents |
|---|---|
| `run_app.py` | GUI launcher |
| `design_tensor.py` | Headless tensor designer |
| `generate_training_data.py` | [Command line data generation](docs/COMMAND_LINE_DATA_GENERATION.md) |
| `materials/` | The six shipped materials |
| `programs/reference_0p1W_burn.json` | The 0.1 W, 34 s reference burn |
| `tools/` | Steady-state RF+DNP optimizer; design plotting |
| `ml/` | Example PyTorch models: [pulse design](docs/TRAINING_PULSE_MODEL.md), [polarization](docs/TRAINING_POLARIZATION_MODEL.md) |
| `ssrf_realtime/` | Model, optimizer, extended physics engine, GUI |
| `docs/` | User guide, materials, physics summary, key results; derivations in `docs/reference/` |
| `tests/` | 460 tests; legacy materials used as fixtures are in `tests/fixtures/` |

Run the tests with:

```bash
python3 -m pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python3 -m pytest -q --import-mode=importlib
```

## What the model is, and isn't

It is a population-rate model: incoherent (weak-drive) RF saturation, conservative spin
diffusion, DNP and T1. The optimizer uses the model's exact adjoint and finds local optima.
[docs/PHYSICS.md](docs/PHYSICS.md) summarizes the equations.

Known limits:

- **DNP is phenomenological.** One spin-temperature term, with rate and saturation fitted to
  build-up data. The solid effect, cross effect and thermal mixing are not modelled
  separately, so there is no dependence on microwave frequency or ESR line in this simplified version.


[docs/KEY_RESULTS.md](docs/KEY_RESULTS.md) lists which mechanisms move Q, and by how much.
