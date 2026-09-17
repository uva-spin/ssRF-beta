# Three-panel display update — validation

## Software scope

The exact input archive was `spin1_ssrf_realtime_tensor_designer_rf_fix.zip`.
Only one existing runtime file was edited: `ssrf_realtime/gui.py`.
Fifteen other ssrf_realtime modules and both launch/command-line scripts remain
byte-for-byte identical. In particular model.py, ideal_model.py, pulse_program.py,
playback.py, tensor_optimizer.py, material_config.py, designer_gui.py and the
manual-profile editor are unchanged. The start/mute/replay and endpoint-pause
method bodies within gui.py are also unchanged (AST comparison).

See SOURCE_INTEGRITY_DISPLAY.json and DISPLAY_CHANGES.patch. Historical source
manifests describe earlier releases, not this GUI-only edit.

## Regression suite

**211 passed, 4 skipped in 34.44 seconds.** Exact output is TEST_RESULTS.txt.
All 189 prior headless tests are retained. Twenty-two display tests were added.
The four existing optional Qt module expectations now allow four trace lines
and four Axes objects (three panels plus the RF right axis).

The new tests execute the ACTUAL shipped GUI plotting/history/export method
bodies with real Matplotlib Agg axes. Only Qt window/widget construction is
replaced by small doubles. Coverage includes:

- Correct three-row order and shared R domain, separate monitor time axis.
- Solid direct traces and dotted blue/yellow mirror traces.
- Literal signed spectral subtraction and integrated-Q consistency for positive,
  zero and negative initial polarization.
- Live population-derived mirror history during burn and RF-off recovery.
- Both components of both packet locations, including M=0.
- Identical display-noise realization for upper branches, total and subtraction.
- Monitor changes/reset, history restart, new material/state adoption and trimming.
- Mirror-aware automatic trace limits; stable zero-signal tensor limits.
- End-of-program pause display and subsequent RF-off recovery.
- CSV column prefix compatibility and accurate appended mirror values.
- Rendering does not advance time, consume pulse exposure or alter populations.
- Accidental plot clicks do not move the monitor or any RF pulse.

## Exact saved-program replay

The original 701-bin starting_state.npz, rf_program.json and validated time step
were loaded unchanged. The same program supplies nonzero exposure to 226 bins
and stops at T=0.19693793171894836.

- P at endpoint: 0.40263731747775855
- Q at endpoint: 0.2249643810837316
- Maximum absolute difference from saved endpoint population array: **0.0**
- Direct and mirror histories: 45 samples each, sharing timestamps.

See outputs/display_validation/replay_and_display_check.json. This checks
software consistency, not material calibration or global optimality.

## Render inspection and limitations

The actual GUI plotting callbacks were rendered through Matplotlib Agg at a
compact 685 x 505-pixel canvas and at higher resolution. The three axes and
labels were visually inspected. The images are plot previews, NOT native Qt
window screenshots; side controls are not rendered by this harness.

No PyQt6, PySide6 or PyQt5 binding is installed in this environment. A PyQt6
installation was attempted and failed because package-host DNS/network access
was unavailable. Native Qt painting, desktop sizing and mouse/signal delivery
were not executed here. The updated optional Qt tests are included for local
execution with an installed binding. Compileall completed for the runtime code.

Commands:

```bash
OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen python3 -m pytest -q
MPLBACKEND=Agg python3 examples/preview_three_plots.py
```
