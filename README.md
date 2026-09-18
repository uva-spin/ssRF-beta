# Real-time ss-RF: tensor manipulation and optimizer

This is a display-only update of the working RF-playback-corrected package.
The main window now has three stacked live plots:

1. **Top:** I+(R), I-(R), and their sum, with the existing monitor/profile markers
   and planned/active RF commands on the right axis.
2. **Middle:** the signed **I+(R) - I-(R)** tensor spectrum, over the same physical-R
   domain as the top plot. A zero line identifies negative/positive regions. The
   monitor M and mirror -M markers follow the same locations in both panels.
3. **Bottom:** absolute intensities versus time at M and -M. Direct I+(M,t) and
   I-(M,t) remain solid. **I+(-M,t) is dotted blue; I-(-M,t) is dotted yellow**
   (a dark golden yellow for visibility on white). No sum or normalization is added.

The main window retains its compact 1000 x 555 default and scrollable controls.
The common frequency-axis label is under the middle plot to avoid duplicate
labels in the compact layout. The lower axis is time, not physical frequency.

All four monitor histories are recorded together from the evolving populations,
including the initial sample. Moving M resets all four to their values at the
new M/-M locations; it does not move RF pulses. Restarting the trace, resetting
populations, or loading a material/state clears all four histories consistently.
RF-off recovery continues updating them unless the simulation is paused.

The middle curve is the literal displayed spectral subtraction, not Q(P), a
Boltzmann fit, a change from a reference, or a local intensity ratio. The separate
live Q readout remains the integrated population-derived tensor polarization.
The top sum and middle difference reuse a single sampled spectrum each refresh,
so they are also consistent when display-only noise is enabled. The time traces
remain noise-free population observables, as the previous direct traces were.

Trace CSV export retains its original first five columns and appends
`mirror_R`, `Iplus_mirror`, and `Iminus_mirror`. Material, snapshot and RF-program
file formats are unchanged.

**The corrected RF ON start/replay behavior and Pause at program end are retained.**
No population equation, recovery parameter, pulse scheduler, RF command, DNP/T1
source, optimizer or material-file schema was changed. See `DISPLAY_UPDATE.md`
and `PLAYBACK_FIX.md` for the current display and playback behavior.

## Corrected RF playback workflow

1. Generate a tensor program and click **Load generated program (do not run)**.
2. Back in the main window, click the RF button labeled **RF OFF — click to start program**.
   That ONE action starts the loaded program at program time zero, turns RF on,
   and resumes the simulation. A separate Start click is no longer necessary.
3. On completion the RF gate returns OFF. Generated programs automatically select
   **Pause at program end** so the immediate endpoint stays visible.
4. Press **Run simulation (continue recovery)** to let the post-pulse line recover.
   Alternatively uncheck **Pause at program end** before a run for uninterrupted evolution.

Running the simulation *before* clicking RF ON is also supported. A READY program
has no running pulse clock and cannot expire while waiting to be started.
The main status line shows READY, RUNNING, WAITING, MUTED or FINISHED, current
program time, active bin count and a delivered-exposure receipt.

RF ON starts or replays a READY, STOPPED or FINISHED program from the current
simulation time and current populations. During an active run it only unmutes;
it does not restart the clock. RF OFF during a run still mutes the source while
the schedule advances. **Pause simulation** freezes both clocks. There is no
automatic repetition and no implicit restoration of the optimization snapshot.

The same RF-button behavior applies to generic examples and JSON/CSV programs.
An empty/disabled pulse table or global gain zero now raises a clear warning
rather than showing an apparently active RF gate. Explicit **Start / restart RF
program** remains available. Loading any program still supplies zero RF until
an explicit start action.

Endpoint hold is a session playback choice, not a material kinetic parameter;
it does not change the existing JSON material or RF-program schemas.

## Install and run

Python 3.9 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run_app.py
```

Windows activation: `.venv\Scripts\activate`.
The existing Qt loader tries PyQt6, PySide6, then PyQt5. For a non-GUI install:

```bash
python3 -m pip install -r requirements-core.txt
```

The new entry point is `run_app.py`, which loads `ssrf_realtime.designer_gui`.
`gui.py` builds the three live panels and retains start/mute/replay and endpoint handling through the unchanged `playback.py`.

## Material configurations

The new panel is **Material configuration / tensor design** in the scrollable
side panel. Use **Load material**, **Save material**, **Save as...**, or
**Edit ALL material / search settings...**. The last opens an explicit JSON
editor, useful for precise values or fields not present in the compact GUI.
Live spin fields remain arrow-only and ignore mouse-wheel changes.

The provided `materials/default_demo.json` is a complete template, **not a
calibrated ND3 parameter set**. Duplicate it for each material. Change its
`name` and `description` as well as the parameter values. Unknown keys, invalid
population triangles, nonfinite or negative rates, and invalid settings are
rejected rather than silently accepted or clamped.

The material file groups:

* grid and analytic lineshape;
* initial P and optional Q (`q0: null` initializes Q from the Boltzmann relation);
* the retained RF coupling and Pake-density correction parameters;
* K0, overlap shape/width, cross-branch and DQ strengths, geometry, cutoff and MW factor;
* DNP saturation and build rate, T1 target and rate, enable flags;
* numerical step, display calibration and noise;
* all tensor-search limits, tolerances, regions and optional per-bin power ceilings;
* GUI steps/tick, trace length, timer interval and window size.

Legacy Voigt fields are also stored for complete parameter round trips, but are
**unused by the ideal RF program**. Pulse powers, pulse gain, pulse timing and
profile selection remain in the separate pulse-program JSON/CSV, not in a
material file. A material is not a runtime checkpoint.

### Applying a file is explicit

Loading or applying material settings stops RF and leaves the simulator paused.
Choose either:

1. **Keep the current populations and simulation time.** This permits rate,
   coupling, reservoir, display and numerical changes without erasing a burn.
   Changed p0/q0 affect the next explicit reset only. A changed grid or static
   lineshape/capacity is rejected in this mode.
2. **Reinitialize explicitly.** This rebuilds the state using the file's grid,
   lineshape and initial P/Q. A compatible planned pulse table is retained;
   an incompatible grid is not silently remapped.

RF is never armed by loading a file, even if the saved RF flag was on. Click **RF ON** (or **Start / restart RF program**) explicitly when ready.

## Generate a tensor program from the current window

1. Load/tune the material. Set initial P/Q, or manipulate the line as desired.
   The generator uses the actual population arrays corresponding to the current
   signal, not the initial P setting, not a fitted equilibrium line and not display noise.
2. Press **Generate tensor program from CURRENT state...**. The live simulation
   and any existing schedule are paused; their state is not advanced by the search.
3. Set the maximum RF command and common-duration search range. The initial
   defaults are U_max=20, T_min=0.01, T_max=2. They are editable trial search
   constraints, not measured RF power or time calibrations.
4. Press **Calculate from captured state**. Inspect progress, predicted endpoint,
   per-bin powers, P/Q changes, and the no-RF comparison.
5. Use **Export JSON / CSV + snapshot...** to write an independent design bundle.
   Or use **Load generated program (do not run)** to put it into the existing
   manual pulse editor. No population or spectrum is overwritten by installation.
6. Back in the main window, click **RF ON** to start and apply it. The simulation
   resumes automatically. At the endpoint it pauses for inspection by default.
7. Press **Run simulation (continue recovery)** to resume RF-off evolution.

The generator uses the current DNP, diffusion and T1 settings throughout the
prediction. It **maximizes positive Q for either sign of P**. The signed
subtraction I+ - I- does not reverse its definition for negative P.

### Numerical validation and playback dt

The selected program is replayed through the unchanged `IdealBinModel`, then
checked against a smaller integration step. If needed, verification halves the
playback dt; it does not change pulse powers, pulse duration, material rates, or
population equations. The validated dt is shown in the result and recorded as
`recommended_playback_dt` in the report. **Loading a generated program from the
designer applies this smaller numerical dt**, if one was needed. The current
live populations remain unchanged. For later JSON/CSV loading or command-line
replay, use the report's dt; `design_tensor.py --replay` does this automatically.

A failed time-step check blocks installation but still permits export for diagnosis.

### What is optimized and what is not

The allowed control is one constant command U_j at each selected physical bin,
all acting on [0,T), with 0 <= U_j <= U_max,j. Both overlapping absorption
components receive the same physical-bin command. Existing packet-coupling
weights remain in effect. U is an effective base equalization-rate scale, **not
calibrated watts**; times and R are the existing simulation units.

The candidate selector defaults to the union of negative signed subtraction
and a positive initial RF tensor slope. A subtraction-only mask, slope-only
mask, or all-bin search is also available. Advanced settings can restrict R
regions or set individual bin ceilings. Bins outside the mask remain off.

Power optimization uses the complete coupled population dynamics. It does not
assign a mirror gain, force all local subtractions positive, reset the state to
Boltzmann, or impose a Voigt hole. For each trial duration, bounded local searches
adjust the bin powers. Among the completed candidates, the shortest endpoint
within the specified tolerance of the best Q found is selected. The report lists
unconverged starts, boundary choices, budget limits and numerical checks.

**This is a finite, bounded local search, not a proof of a global maximum or
of the globally shortest possible pulse.** Search limits and available
mechanisms determine what can be found. Longer, stronger or different pulse
families are not silently considered. No-RF predictions distinguish RF benefit
from ordinary DNP buildup or tensor equilibration.

The optimization can take seconds to minutes depending on bin count, rate
stiffness, duration range and search settings. It runs in a cancellable worker;
the main live state remains frozen. The wall-time budget covers the search;
endpoint validation is performed afterward and is also cancellable.

## Export contents

Each export creates a **new** `tensor_design[_NN]` directory containing:

| File | Content |
|---|---|
| `rf_program.json` | Existing loadable pulse schema; all active starts 0, duration T, gain 1 |
| `rf_program.csv` | Existing long-form per-bin pulse table |
| `material.json` | Source-state material settings and exact search settings |
| `starting_state.npz` | Current n, reference n, capacities, grid, source settings and fingerprint |
| `optimization_report.json` | Objective, limits, P/Q prediction, no-RF endpoint, warnings, validated dt |
| `duration_search.csv` | Every completed trial horizon and search outcome |
| `bin_diagnostics.csv` | Powers/exposures and initial/final branch intensities for every bin |
| `predicted_PQ_trace.csv` | Full-model replay through the shared endpoint |
| `predicted_endpoint.npz` | Final and no-RF population arrays |

A changed live state requires regeneration. Loading `starting_state.npz`
explicitly restores that state and its material parameters; it does not restore
an active pulse or automatically run anything. An arbitrary experimental trace
is not loaded by this feature: its two branch populations must first be
reconstructed separately.

## Headless use

```bash
# Initialize a saved material and design a program
python3 design_tensor.py --material materials/default_demo.json --out my_design

# Design from a saved manipulated state
python3 design_tensor.py --state my_state.npz --out new_design

# Override selected search limits
python3 design_tensor.py --state my_state.npz --max-power 10 --duration-max 1 --out another_design

# Reproduce the exported endpoint using the exact pulse scheduler
python3 design_tensor.py --replay my_design

# Start the GUI at a material's initial state, paused
python3 run_app.py --material materials/default_demo.json

# Start the GUI at an explicit saved state, paused
python3 run_app.py --state my_state.npz
```

Output directories must be new to avoid overwriting previous designs. State and
material files with different parameters are not silently combined.

## Included package

`TENSOR_DESIGN.md` describes the optimizer and its assumptions.
`USER_GUIDE.md` contains the detailed manual-profile editing instructions.
`RECOVERY_MODEL.md`, `DISPLAY_UPDATE.md`, and `PLAYBACK_FIX.md` document the
current model, display, and RF playback behavior. The package intentionally
omits development-only tests, provenance archives, patch files, and generated
output directories.
