# Display update

Three live panels now show the full absorption spectrum, its signed tensor
subtraction I+ - I-, and the monitor time traces. The time panel includes solid
I+(M), I-(M), dotted blue I+(-M), and dotted yellow I-(-M). Monitor controls move
M/-M in both spectral panels without moving RF commands. All histories reset
together when M changes. CSV export now includes the mirror intensities.
See README.md and DISPLAY_UPDATE.md for the current display details. The manual
pulse-program controls below continue to work without changes.

---

# Playback update

This release fixes the RF button: after loading a pulse program, **RF ON starts
it from program time zero**. It also resumes the simulation. A READY program
does not run out while RF is off. A FINISHED program is replayed only after an
explicit new click. No populations are reset when starting or replaying.

**Pause at program end** holds the final signal; it is selected automatically
when a newly generated tensor program is loaded. Run simulation then resumes
recovery without applying more RF. While a schedule is already active, RF OFF
still mutes it without stopping its clock; Pause simulation freezes both.

The detailed manual-table controls below are unchanged. Older instructions to
press the separate Start button remain valid, but that extra step is no longer
required when using RF ON. See PLAYBACK_FIX.md for the complete state table.

---

# Real-time ss-RF: custom per-bin RF powers and timing

This is a separate **manual-control** branch of the working spin-1 simulator.
You define the power/rate and application time in each physical frequency bin.
The software applies that schedule to the three-level population equations.
It does not choose an optimum, impose a hole shape, or add Voigt spreading.

## Run

Python 3.9 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run_app.py
```

Windows activation is `.venv\Scripts\activate`. The interface tries PyQt6,
PySide6, and PyQt5 in that order. An existing environment with NumPy, Matplotlib
and one of these Qt bindings can run `python3 run_app.py` directly.

For a different Qt binding, install `requirements-core.txt` and that binding
instead of the PyQt6 requirements. The ideal model does not require SciPy;
SciPy is needed by some of the reference/regression tests.

## First run: enter your own value in EVERY bin

1. Click **Edit EVERY bin: power / start / duration** in the main window.
2. Click **Add custom profile**. Set **Left L (R)** and **Right R (R)** with
   their on-screen arrows, or use **Enter bounds...** for a deliberate exact
   numerical entry. Bounds snap to actual bin centers; the row count and bin
   indices are displayed underneath.
3. In the default **Per-bin numerical values** tab, click
   **Add every missing bin in L..R (U=0)**. This gives you one row for each bin
   in the range. New rows start at zero RF power; existing rows are preserved.
4. Every row has its OWN numerical controls for **RF power U**, **Start time**,
   and **Duration**. Use its arrows to adjust it. To type exact numbers, select
   the row and click **Enter exact values for selected rows...**, enter the
   numbers, and confirm **Apply exact values**. Stop time is calculated for you.
5. Repeat for any bins you choose. For bulk changes, select several rows, click
   the exact-values button, and check ONLY the properties you want to set.
   Unchecked values stay unchanged. Multiple-row dialogs begin with no properties
   checked, preventing accidental overwriting of differing durations or powers.
6. Add other profiles for other spectral regions or pulse sequences. Duplicating
   rows copies their start times too: set later start times for repeated pulses.
   Otherwise simultaneous duplicate commands add, exactly as before.
7. Click **Apply program**, then **Start / restart program** in the main window.

All of this is manual numerical control; there is no optimizer or forced shape.
Zero power, zero duration, or an unchecked On box gives no irradiation.
The power column remains the model's effective RF base rate, not calibrated watts.

### Custom table entry / copy and paste

**Paste custom table...** accepts comma-, tab-, or space-separated values:

```csv
bin_index,rate,start,duration
240,1.35,0.00,0.25
241,2.10,0.00,0.40
242,0.80,0.10,0.30
```

Each row can have completely unrelated numerical values. To enter physical
frequency instead of a bin number, use an `R,rate,start,duration` header; R is
snapped to the nearest bin. An optional fifth `enabled` column accepts 0/1 or
true/false. **Copy table** includes this fifth column so disabled pulses remain
 disabled when pasted. Repeated rows for the same bin are allowed.

The paste dialog replaces ONLY the selected profile after you confirm it.
Malformed rows are rejected; it does not partly apply a failed paste. Existing
other profiles are not changed. The main simulation still receives no changes
until **Apply program** is pressed.

`examples/custom_bin_values.csv` is a compact table for this paste dialog.
`examples/custom_bin_values_program.json` is the corresponding loadable program.
The compact four/five-column table is NOT the metadata-rich CSV accepted by the
main **Load program** action; use Paste custom table for the compact format.

### Optional presets

The **Optional shape presets** tab still provides flat, ramp, triangle, and
bounded Gaussian command generators. They are conveniences only. Replacing a
nonempty profile with a preset now asks for confirmation. The generated rows
appear back in the numerical table, where every power and time remains editable.
**Load horn / pedestal example** also asks before replacing the editor program.
No example is claimed to maximize tensor polarization.

## Safe controls: arrows versus explicit numerical entry

- Numeric spin fields are **arrow-only**. Neither mouse-wheel/trackpad scrolling,
  typing into their displayed text, nor keyboard arrow stepping changes them.
  Click their visible up/down buttons for incremental changes.
- This protection applies to the main RF/DNP/diffusion/numerical controls AND
  to the individual bin controls in the pulse editor. Closed dropdowns do not
  change selection when scrolled over either.
- Exact typed input is available through the labeled **Enter exact values...**,
  **Enter bounds...**, and **Enter exact monitor R...** dialogs. These are explicit
  actions and require Apply; simply focusing or scrolling a field does not edit it.
- Ordinary clicks on the spectrum no longer move any marker. The marker controls
  below provide a clear, deliberate way to reposition the readout.

## Which markers are which?

| Marker / overlay | Meaning | How to set it |
|---|---|---|
| **M**, dashed monitor line and direct points | Location of I+ and I- shown in the lower time plot; NOT an RF carrier setting | **Marker positions**: use **Monitor bin M** arrows or **Monitor R (M)** arrows. **Enter exact monitor R...** accepts a numerical R and snaps to the nearest bin. |
| **-M**, dotted mirror line and opposite points | Mirror frequency of the monitoring bin | Automatically follows at minus the actual monitor R. Its readout is intentionally read-only. |
| **L / R**, dash-dot profile limits | First and last CONFIGURED bin of the selected RF profile | Choose a profile in **Show RF-profile limits L / R**. Change its actual pulse bins in the editor and Apply program. |
| **Program envelope** | Maximum scheduled combined command in each bin, over the whole program | Comes from the bin pulse tables. It is not continuous irradiation. |
| **RF applied now** | Actual combined RF command at the current program time | Comes from each pulse's power/start/duration and the RF ON/OFF gate. |

Changing M starts a new lower history at the newly selected bin's current
intensities. It does NOT change populations, the RF program, delivered exposure,
or the program clock. The mirror cannot be independently moved because it
represents the same packet's other transition.

The editor's L/R entry fields select a range for creating rows. They do NOT
stretch, move, overwrite, or delete existing custom pulse values. Add missing
bins or explicitly edit/delete pulse rows as needed. On the main plot, L/R show
the bounds of the committed profile, including disabled/zero-power rows. Gaps
between those bounds are not silently filled. Choose **Hide profile bounds**
when only the monitor/mirror markers are needed.

## What each command means

A row represents

```
physical bin j, base RF rate U, start s, duration d
```

and contributes `U` in that bin during the half-open interval `[s, s+d)`.
Times are relative to the last press of **Start / restart program** and are in
**simulation-time units**, not wall-clock seconds until the model is calibrated.
The command's spectral support is exactly its chosen bin. No adjacent-bin RF
leakage, smoothing, automatic normalization, or convolution is applied.

The shape generator is only an editor convenience: it writes explicit numbers
into the table. Triangle/Gaussian **Rate / peak U** is the largest generated bin
value. A Gaussian command is cut off at the selected region's end bins, with
exactly zero commands elsewhere; it is not a material-response kernel. A linear
ramp uses Rate/peak U at the left and Right rate at the right. Bounds snap to
nearest grid centers inclusively; inspect the table for the actual coordinates.

**Start step / bin** staggers neighboring start times. For a sequential sweep,
set it equal to a common pulse duration; for simultaneous irradiation set it
zero. **Duration left/right** linearly initializes durations across the region.
Every generated value can then be edited individually.

### Overlapping profiles

All enabled contributions to the same physical bin and time **add**. If one
profile commands U=2 and another U=3 there, the combined command is U=5. No
clipping or renormalization is hidden. The editor reports the largest combined
command. Nonoverlapping pulses on the same bin remain separate in time.

This is an ideal effective-rate model; additive rates are not a coherent
multitone phase or interference calculation, and no total hardware power limit
is imposed in this first manual-control version.

### Common field and inherited coupling

There is ONE commanded value `u[j]` at a physical R bin, shared by both
absorption branches at that frequency. There are no separate user powers for
I+ and I- at the same physical R.

The working model's packet convention is preserved:

```
packet k: I+ at x[k], I- at -x[k]
Gamma_plus[k]  = w[k] * u[k]
Gamma_minus[k] = w[k] * u[N-1-k]
```

`u` includes the explicit **global RF gain**, default 1. The optional `w[k]`
is the same phenomenological Pake-density coupling weight retained in the
working package. The population equations are unchanged:

```
J_plus  = Gamma_plus  * (n_plus - n_zero)
J_minus = Gamma_minus * (n_zero - n_minus)
dn_plus  = -J_plus
dn_zero  =  J_plus - J_minus
dn_minus =  J_minus
```

The existing diffusion, DNP and T1 derivatives are added to these. All bins
are evolved together, including cases where both transitions of a packet are
addressed. The mirror intensities come only from the populations. No mirror
area or gain is assigned separately.

**Power units:** U uses the previous effective base equalization-rate scale,
not calibrated watts or B1. With the default packet weighting, the actual
transition coefficients are `w[k]*U`, and the current effective coefficients
are shown in the side panel. Setting **rate-density power=0** gives `w[k]=1`;
as in the baseline, that existing knob also affects DNP weighting. No extra
capacity multiplier was added by the pulse scheduler.

## Time controls and on/off behavior

| Control | Result |
|---|---|
| Start / restart program | Sets program time to zero at the current simulation time and enables the RF master gate. Populations are not reset. It also restarts the lower history from the current monitored intensities. |
| RF OFF | Mutes the RF contribution only. Both simulation time and program time continue. Pulse intervals that pass while muted are not replayed automatically. |
| RF ON | Re-enables the RF gate for whatever pulses are scheduled now. It does not start an idle or completed program; use Start / restart program. |
| Pause simulation | Freezes population evolution and program time together. Resume continues from that state. |
| Stop program | Cancels scheduled output and turns RF off, without resetting populations or stopping recovery. |
| Last pulse ends | Scheduled RF becomes exactly zero. Diffusion, DNP and T1 continue according to their switches. |
| Reset populations | Rebuilds the initial population state and resets simulation time. Retains the current profile definitions but stops/disarms their execution. |
| Apply/load a program | Validates and installs a copy, stops scheduled RF, and preserves current populations. Press Start to execute it. |

Editing/loading uses a frozen simulation clock while the dialog is open, so a
pulse does not quietly run out during editing. Cancel leaves the program and
its elapsed time unchanged; Apply installs a stopped program.

Pulse start/stop events split the numerical steps. Durations are not rounded
to the GUI timer interval or to the nominal `dt`, including sub-dt pulses.
This makes scheduling exact to floating-point time resolution; it does **not**
make the numerical population solution exact. The population integrator remains
forward Euler, with step subdivision when required for stability and a failure
rather than silent clipping of a significant negative population. Convergence
in `dt` still matters.

The Qt timer drives screen updates only. Qt timers can be delayed by the
operating system or a busy event loop; therefore they are not used as physical
pulse clocks. Reference: Qt documentation, *QTimer / Accuracy and Timer
Resolution*, https://doc.qt.io/qt-6/qtimer.html .

## Two main dynamic plots

The top panel keeps the working Pake spectrum and its direct/mirror markers.
Its right axis adds two **base-rate** overlays:

- **Program envelope:** the maximum scheduled command at each bin over the
  whole program (a planning aid, not a continuously applied profile).
- **RF applied now:** the actual combined command after the RF master gate.

The lower panel contains only absolute **I+(R,t)** and **I-(R,t)** at a selected
**monitor R**. Changing the monitor using its controls starts a new local
history but does **not** move the pulse profiles or reset program time.
Spectrum clicks have no selection effect. This
separation is necessary when several regions are driven simultaneously.

The main window is approximately 1000 x 555 pixels, reduced when necessary to
fit the available screen. Material controls remain in a scrollable side panel;
the larger table is in a separate editor window.

## Save, edit, and repeat programs

Use JSON to preserve all profile metadata, including empty/disabled profiles,
grid information and global gain. The long-form CSV is useful for editing large
per-bin schedules outside the GUI. CSV columns are

```
profile_id,profile,profile_enabled,enabled,bin_index,R,rate,start,duration,n_bins,r_min,r_max,gain
```

There can be multiple rows for a bin. The R column must match its bin index;
programs are rejected if their grid differs from the model grid. The software
does not silently interpolate an ideal command onto a different grid. Empty
profiles have no rows and therefore are omitted from CSV; JSON preserves them.

A saved program describes RF settings, not a serialized nuclear state or a full
material parameter configuration. Use the same P0, diffusion, DNP and coupling
settings when comparing repeated runs.

**Export trace / exposure CSV** saves the two displayed monitor traces and a
second file with delivered per-bin base exposure `integral u[j](t) dt`. Exposure
is recorded after the RF gate and global gain, before the packet coupling
weight; therefore it audits the command actually delivered. The file separates
exposure since the latest program start from exposure since the last population
reset. Changing a command mid-run is reflected in delivered exposure.

## Headless / scripted use

```python
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.pulse_program import PulseProgram, RFProfile, BinPulse, make_profile

m = IdealBinModel(IdealBinParams(p0=0.45))
p = PulseProgram(len(m.Rplus), float(m.Rplus[0]), float(m.Rplus[-1]))
p.profiles = [
    make_profile(m.Rplus, "horn", -0.98, -0.72,
                 shape="triangle", rate=2.0, start=0.5, duration=0.5),
    make_profile(m.Rplus, "pedestal", 1.15, 1.65,
                 shape="linear", rate=1.0, rate_right=3.0,
                 start=0.8, duration=1.0, duration_right=1.8),
    RFProfile("individual pulses", pulses=[
        BinPulse(bin_index=150, rate=0.8, start=3.0, duration=0.2),
        BinPulse(bin_index=151, rate=1.4, start=3.1, duration=0.35),
    ]),
]
p.save("my_program.json")
m.set_program(p)
m.start_program()
m.step(n_steps=2000)
print(m.polarizations())
```

The program and model must share the same grid. The default grid has 701 bins
from -3 to +3, with 0-based indexing. `gamma_rf` in the retained baseline model
is **not** a second multiplier on scheduled commands. In this branch, use each
pulse's `rate` and `model.set_program_gain(...)`.

Run the demonstration with

```bash
python3 examples/headless_demo.py
```

It produces an editable horn/pedestal program, command-profile diagnostics,
local intensity traces and a delivered-exposure audit. It contains no automatic
tensor maximization.

## Validation and provenance

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
# With a Qt binding installed, optional window/editor smoke test:
QT_QPA_PLATFORM=offscreen python3 -m pytest -q tests/test_gui_optional.py tests/test_arrow_gui_optional.py
```

Tests include multiple profiles and overlapping pulses, start/stop edge handling,
sub-dt durations, per-bin exposure, mirror indexing, full three-level RF matrix
rates, source conservation, negative polarization, JSON/CSV round trips, direct
regression to the working one-bin package, and comparisons with the RF-only
matrix exponential. The old 21 tests are retained as baseline regressions.

The baseline `model.py`, `lineshape.py`, `rf_profile.py` and `baseline_gui.py`
are byte-identical to the working archive. `BASELINE_MANIFEST.json` records
hashes. New functionality is in `ideal_model.py`, `pulse_program.py`,
`profile_editor.py`, and the new `gui.py`. Legacy Voigt functions are retained
only for compatibility/regression; the ideal RF path does not call them.

The inherited diffusion/DNP/extra packet-weighting parameters remain
phenomenological. Passing software tests demonstrates implementation consistency,
not experimental validation or global tensor optimality.

## Scope of this update

This update changes only the interface and pure editing helpers. The population
model, ideal RF scheduler and time stepping, pulse-program schema, static
lineshape, RF/DNP coupling weights, diffusion kernel, and all material defaults
are unchanged. `SOURCE_INTEGRITY.json` contains hashes compared directly with the
previous ideal-profile ZIP, and a test enforces them. Existing JSON programs and
full-format CSV programs remain compatible.

The underlying design uses Qt's documented QAbstractSpinBox/QLineEdit separation:
only the embedded line edit is made read-only, leaving the on-screen spin buttons
usable. Wheel events are ignored. Implementation references:
https://doc.qt.io/qt-6/qabstractspinbox.html
https://doc.qt.io/qt-6/qlineedit.html

See `VALIDATION.md` for the exact executed tests and the limits of GUI validation.
