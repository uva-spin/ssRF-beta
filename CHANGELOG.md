# Display-only update: middle tensor spectrum and mirror monitor traces

- Three compact panels: spectrum, I+ - I- over the same R domain, time traces.
- Four synchronized absolute monitor histories; dotted blue/yellow mirrors.
- Single shared spectrum sample for top sum and middle difference.
- Mirror columns appended to monitor CSV; existing five-column prefix preserved.
- No physics, RF, optimizer, material, or playback behavior changes.

---

# RF playback correction

- RF ON now starts a READY/STOPPED program or explicitly replays a FINISHED one.
- Loading does not apply RF; running the simulation cannot consume a READY program.
- An active schedule is unmuted without resetting its time or delivered exposure.
- Completion turns the RF indicator off and reports the delivered exposure.
- Generated programs enable optional endpoint pause for immediate Q inspection.
- Empty programs and zero global gain produce an explicit warning.
- RF-checkbox updates block signals so completion/loading cannot retrigger pulses.
- Core model, optimizer, scheduler, file schemas and material values are unchanged.

See PLAYBACK_FIX.md for precise playback behavior.

---

# Material library and synchronized tensor-profile design

Added material JSON persistence and independent population snapshots. The entire
IdealBinParams dataclass, including backend-only values, is grouped and stored.
New files are versioned, validated, and saved atomically. Loading cannot silently
regrid a manipulated state or energize RF.

Added a cancellable tensor designer from the current population state. It uses
simultaneous constant bin powers, one common stop time, bounded duration/power
search and immediate endpoint Q. The existing material kinetics stay active
according to captured switches. A numerical gradient differentiates those same
equations; the final result is independently replayed using the existing pulse
scheduler and checked at finer dt. Export writes the existing JSON/CSV pulse
formats plus snapshot, material, convergence report and per-bin diagnostics.

The main live GUI retains the main spectrum, tensor-spectrum, and time-trace
plots. Manual per-bin editing,
protected numeric fields, independent monitor placement and all recovery/RF
controls remain. The app entry point loads `designer_gui.py`.

No physical rate values, RF coupling law, mirror response, DNP source, T1 term,
or recovery mechanism was changed. Verification can recommend a smaller
numerical playback dt; this is shown and applied only when the generated program
is explicitly loaded from the designer.
