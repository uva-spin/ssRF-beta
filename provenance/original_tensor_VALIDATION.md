# Validation of material persistence and synchronized tensor design

Executed in the supplied runtime with NumPy, SciPy and Matplotlib; no Qt binding
was available. Command:

```text
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg pytest -q
165 passed, 3 skipped
```

The three skipped items are optional Qt test modules: the earlier manual-editor
and arrow-only GUI tests, and the new material/designer-window tests. GUI source
compilation succeeds, but the live window and QThread event flow were NOT
executed in this runtime. Local commands to exercise them are included in tests.
No graphical test is counted as passed on the basis of source compilation.

## Preserved implementation

All 12 inherited Python source files under ssrf_realtime are byte-identical to
`spin1_ssrf_realtime_ideal_profiles_boltzmann.zip`. See
SOURCE_INTEGRITY_TENSOR.json for archive and file SHA-256 hashes. The run_app.py
entry point selects the new workflow subclass. All inherited kinetic regression
tests are retained unchanged.

## New tests

* Complete material parameter coverage and JSON round trips.
* Strict invalid-value and unknown-key rejection.
* Preservation of manipulated populations, reference arrays and time on compatible material changes.
* Refusal to silently change grid/capacity during preserve-state loading.
* Explicit initialization from changed initial P/Q.
* Independent snapshot round trips, fingerprint checks and detection of changed local states even when total P is unchanged.
* Algebraic forward RHS agreement with unchanged model.py for random non-Boltzmann states, both overlap shapes, cutoff choices, orientation correlation, DNP, T1, cross-branch and DQ exchange.
* Analytic state/control Jacobian-transpose products versus finite differences.
* Full time-discrete endpoint gradients versus finite differences.
* Correct signed candidate geometry for negative P.
* Independence from display noise and display calibration.
* Optional region restrictions and individual bin ceilings.
* Synchronized bounded commands and exact pulse exposure accounting.
* Optimization from a non-Boltzmann current state, not a reinitialized line.
* Single-bin RF-only optimizer agreement with its analytical exposure optimum.
* No damaging pulse forced on a state already at maximal tensor polarization.
* Negative-P enhancement, cancellation and unchanged live input arrays.
* Exported JSON/CSV round trips and independent program replay.

## 701-bin demonstration

File: outputs/default_701_design/optimization_report.json.
The model uses the inherited default Boltzmann-recovery parameters, initial
P=+0.45, DNP off and T1=0. Search ceiling U=20, duration range 0.01–2,
7 logarithmic horizons, 3 refinements, 2 starts and 30 local iterations/start.
The demonstration used a 150-second search budget; the GUI default budget is
300 seconds. The budget was not reached.

| Quantity | Value |
|---|---:|
| Initial P | 0.45000000000000007 |
| Initial Q | 0.15812595436061372 |
| Common duration | 0.19693793171894836 |
| Active bins | 226 |
| Largest commanded U | 20.0 |
| Predicted final P | 0.4026373174777582 |
| Predicted final Q | 0.22496438108373198 |
| Validated playback dt | 0.000375 |
| Q difference against half-step check | 1.6810690572577336e-5 |
| Weighted fractional-population RMS difference | 1.894007300524198e-5 |
| Independent exported-program replay discrepancy | 0.0 (all populations) |

The common duration is the shortest near-best candidate found in this finite
search, not a global shortest-time proof. Several local starts reached their
iteration limit; the report preserves those warnings rather than describing
every start as converged. Rates remain uncalibrated demonstration parameters.
The example is NOT a quantitatively validated ND3 prediction.

Replay:

```bash
python3 design_tensor.py --replay outputs/default_701_design
```

No live state is overwritten by prediction. DNP and no-RF endpoint comparisons
are included in every generated report so natural buildup is not assigned to RF.
