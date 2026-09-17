# RF playback correction — validation

## Verified scope

The input is the exact previously supplied `spin1_ssrf_realtime_tensor_designer.zip`.
Only two runtime source files were edited: gui.py and designer_gui.py. A small
Qt-independent playback.py module was added. Thirteen other runtime source
files are byte-for-byte identical, including model.py, ideal_model.py,
pulse_program.py, tensor_optimizer.py, material_config.py, the lineshape and
manual editor. See SOURCE_INTEGRITY_PLAYBACK.json and RF_PLAYBACK_CHANGES.patch.
No optimization objective, physical rate, population equation or pulse shape
was adjusted to make the result more visible.

## Reproduced original failure

`outputs/playback_validation/original_noop_reproduction.json` records the original
load -> run simulation -> gate ON behavior. For both the default one-bin example
and the generated 701-bin program, program state stayed READY, the RF gate was
true, active command bins were zero, and delivered exposure was zero. Starting
the program explicitly made the original numerical RF operator act normally.

The horn/pedestal example also remained READY under gate-only activation. After
an explicit start it correctly enters WAITING during its scheduled initial delay.

## New tests

`test_playback.py` exercises the real scheduler and population model: fresh starts
at nonzero simulation time, generic/JSON/CSV program loading, leading delays,
finite delivered exposure, no accidental start during loading, mute/unmute,
replay after completion, empty/zero-gain refusal, exact endpoint hold, and
unchanged trajectories when continuous recovery is selected.

`test_playback_gui_callbacks_headless.py` executes the ACTUAL callback bodies
compiled from the GUI source with simple widget doubles. It reproduces generated
installation -> Run simulation -> RF ON, signal-blocked synchronization,
material-state preservation, endpoint hold, replay, and explicit restart. This
is a control-flow test, not a Qt rendering or native mouse-event test.

`test_playback_gui_optional.py` supplies real Qt button and designer-worker tests.
Those can be run locally with an installed Qt binding and QT_QPA_PLATFORM=offscreen.

## Native Qt limitation

No PyQt6, PySide6 or PyQt5 was installed in the execution environment. Installation
was attempted but failed because package-host DNS/network access was unavailable.
The live Qt window was NOT exercised here. Source compilation and callback
harness tests are not reported as native Qt window validation. The optional Qt
test modules are explicitly skipped in the recorded suite result.

## Saved 701-bin generated program replay

The saved starting_state.npz, material parameters, rf_program.json and validated
dt=0.000375 were used without changes. RF ON started the schedule and delivered
nonzero RF to all 226 active bins. With endpoint hold selected, the controller
paused at T=0.19693793171894836 with RF off.

| Check | Result |
|---|---:|
| Initial P | 0.4499999999999989 |
| Initial Q | 0.1581259543606130 |
| Endpoint P | 0.4026373174777586 |
| Endpoint Q | 0.2249643810837316 |
| Bins receiving nonzero exposure | 226 |
| Maximum population-array difference from the prior exported prediction | 0.0 |

See outputs/playback_validation/generated_program_replay.json. This verifies
software playback consistency, not material calibration or physical optimality.

## Regression suite

The exact recorded command and outcome are in TEST_RESULTS.txt. The inherited
physics, optimizer, material-persistence, timing and editor tests are all kept.
To run all tests locally:

```bash
OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen python3 -m pytest -q
```

For only the native playback-window tests:

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest -q tests/test_playback_gui_optional.py
```

Historical tensor-designer validation is preserved in
provenance/original_tensor_VALIDATION.md; its GUI-hash claims describe that older
release, not the two GUI files modified in this correction.
