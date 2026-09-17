# Live tensor spectrum and mirrored monitor histories

## Scope

Only `ssrf_realtime/gui.py` changes among the existing runtime Python modules.
Its plot layout/redraw and trace recording/export are extended. The existing
`designer_gui.py`, `playback.py`, models, optimizer, material configuration,
pulse scheduler, manual editor and safe widgets are unchanged.

There are now three visible plot panels, not four: the fourth Matplotlib Axes
object is the existing RF command right axis overlaid on the top panel.

## Definitions

At selected physical monitor M, the recorded population observables are:

- I+(M,t) — existing solid direct + trace.
- I-(M,t) — existing solid direct - trace.
- I+(-M,t) — dotted blue mirror + trace.
- I-(-M,t) — dotted dark-yellow/goldenrod mirror - trace.

The code calls the existing `pair_intensities(M)` accessor. It does not evolve
separate mirror states, assign any area gain, or synthesize mirrors from a
fixed relation to the direct holes. All four histories use the same timestamps
and trim to the same configured history length. At M=0 the corresponding
mirror/direct traces coincide; this is intentional.

The middle panel plots I+(R,t)-I-(R,t) on the displayed physical-R grid. No extra
calibration, refit, normalization or Boltzmann constraint is applied. In a
noise-free frame, its discrete integral times dR/display_cal agrees with the
existing population-derived Q, within roundoff. It is the spectral contribution
to Q, not the dimensionless total Q at every R.

A single call to `spectrum()` supplies all upper component arrays and the middle
subtraction. This avoids subtracting independently sampled display-noise frames.
The top total is also the exact sum of the displayed branches. Display-only
noise still never enters population equations or optimization.

## Marker and playback behavior

Use Monitor bin M / Monitor R arrows or Enter exact monitor R to position M.
-M follows automatically. Changing M changes monitoring only. Profile pulses
continue at their programmed bins. The previous protection from accidental
plot clicks and mouse-wheel edits is retained.

RF ON still starts a READY program, replays a FINISHED program explicitly, or
unmutes a running program. Loading does not apply RF. Pause at program end holds
all three views at the final pulse endpoint. Continuing the simulation advances
RF-off recovery in all three panels. These playback callbacks are unchanged.

## File compatibility

Existing material JSON, state NPZ, program JSON/CSV and optimizer exports remain
loadable without conversion. The monitor-history CSV alone gains three columns:

    time_since_selection,simulation_time,monitor_R,Iplus,Iminus,mirror_R,Iplus_mirror,Iminus_mirror

The first five columns preserve the previous names and order. Exposure CSV
export remains unchanged. No extra files are required to run the application.

## Display implementation note

The layout uses Matplotlib Figure/GridSpec shared axes inside the existing Qt
window. These are GUI implementation details, not support for any material model.
