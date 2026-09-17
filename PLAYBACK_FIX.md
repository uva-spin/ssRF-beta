# RF start/load/replay correction

## Reproduced cause

The previous GUI wired RF ON to `model.set_rf_enabled(True)` only. Meanwhile,
loading called `model.set_program`, which intentionally clears `_epoch` and
leaves the program READY. `commanded_rf_field()` returns zero when `_epoch` is
None. Consequently, loading -> Run simulation -> RF ON left the gate enabled
but the actual RF command vector zero. No exposure was delivered. The generic
single-bin example had the same issue.

This was a UI/scheduler orchestration bug. It was NOT a problem in the RF
population equations, optimization result, pulse powers or material rates.
In the unpatched release the workaround is the separate Start / restart program
button. That button starts the scheduler and enables the RF gate together.

## Corrected control state table

| State before clicking RF ON | Result |
|---|---|
| READY (newly loaded) | Start at the current simulation time, program elapsed time 0, RF enabled |
| STOPPED | Start again from the current populations, program elapsed time 0 |
| FINISHED | Explicitly replay once from the current populations, elapsed time 0 |
| RUNNING/WAITING but muted | Unmute the existing schedule; do not restart it |
| Empty/disabled pulse program | Warn; RF remains off |
| Global gain = 0 | Warn; RF remains off |

The explicit Start / restart RF program button always starts a new run. Starting
and replaying do not reset the populations, gain, material settings, source-state
snapshot or integrated polarization.

File loading alone continues to apply NO RF. Running the simulation before
starting a READY program does not consume any pulse duration. After a one-shot
program ends, the displayed gate is turned off and FINISHED is shown. No automatic
repeat occurs. The completion receipt shows exposed bins and delivered sum(U dt).

An intentionally delayed program is reported as WAITING, not as a broken active
pulse. During an already-started run, RF OFF remains a mute: if the simulation
continues then scheduled time advances. Pause simulation is the way to freeze
both clocks. A program which finishes while muted is clearly marked FINISHED
and may be replayed explicitly.

## Exact endpoint inspection

`Pause at program end` is enabled when installing a generated tensor program.
The GUI then advances to the end of the final pulse and pauses before subsequent
recovery. It calls the same population stepper, shortening only the last step
when necessary to stop at that endpoint. The original numerical dt is restored.
No physical rate or pulse duration is adjusted, and no extra saturation rule or
mirror-gain assignment is introduced.

Run simulation continues recovery after this pause with RF off. Unchecking the
option retains continuous post-pulse evolution. The checkbox is a session
playback preference, not a material parameter and not an RF-program-file field.
The manual/generic-program default is continuous evolution, as before.

## Qt signal safety

Programmatic synchronization of the RF checkbox blocks its signals. It cannot
start a second run when a loaded program disables RF or when a program finishes.
Qt documents that `setChecked` can emit `toggled`, unlike `clicked`, which is
emitted only by activation (or `click`/`animateClick`). Reference:
https://doc.qt.io/qt-6/qabstractbutton.html#toggled

The added headless callback harness uses the actual GUI method bodies with small
widget doubles. It tests control flow but is NOT a native Qt mouse/rendering test.
The optional Qt regression module exercises actual buttons and the designer
worker on a system with an installed Qt binding.

## Scope

Changed runtime files: `ssrf_realtime/gui.py`, `ssrf_realtime/designer_gui.py`.
Added runtime file: `ssrf_realtime/playback.py`.

Unchanged: population equations and model, ideal scheduler, pulse schema,
lineshape, optimizer, material configuration and snapshot logic, manual editor,
arrow-only widgets and kinetic defaults. See SOURCE_INTEGRITY_PLAYBACK.json.

Use a fresh extraction directory to avoid importing mixed versions. Saved
material configurations, population snapshots and pulse-program JSON/CSV remain
compatible. To reproduce a saved prediction, first explicitly restore its
starting_state.npz and original material settings; a delayed start after DNP or
recovery has altered the line can give a different endpoint.
