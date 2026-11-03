> Development-era derivation reference. Version names (MaterialTuning-N) refer to the history that produced ssRF; file paths mentioned here may not exist in this distribution. See docs/PHYSICS.md for the current summary.

# Total RF exposure budget

MaterialTuning-3 also supports an independent instantaneous total-power ceiling,
`OptimizerSettings.max_total_power`. For the synchronized pulse it enforces
`sum(U_j) <= max_total_power` at every searched duration. `null` is unlimited;
zero forbids RF. The GUI exposes **Limit total RF power** next to the exposure
budget. A duration T must satisfy both constraints, using the tighter of
`max_total_exposure` and `T * max_total_power` as its exposure allowance. Exported
reports record and check each limit separately. Old files migrate to unlimited
total power, preserving their previous behavior.

In `materials/experimental_event_1406922415.json`, total U=1 is normalized to
the user's 0.1 W irradiation. Its power ceiling is 1 and exposure ceiling is 34.
If event identifiers measure seconds, this is a 0.1 W power limit and 3.4 J
nominal incident RF energy. This experiment-specific conversion is provisional;
other demonstration presets do not acquire that calibration. Manual pulse-table
edits and RF gain remain under user control and are not hardware interlocks.

This version was copied from `withAFP/ssRF-beta` after its AFP material/snapshot
compatibility fix. It adds an optional optimizer constraint without changing
the population dynamics, AFP operation or pulse playback.

## Meaning of the setting

For a synchronized sequence with physical-bin commands U_j and duration T:

    E_RF = sum_j integral U_j(t) dt = T * sum_j U_j <= B.

`OptimizerSettings.max_total_exposure` is B. `null` means unlimited; zero forbids
RF commands. Values must be finite and nonnegative. The existing per-bin maximum
and any individual ceilings remain in force. The optimizer need not spend the
entire budget when more exposure does not improve Q.

Each physical bin counts once, with no extra factor for its two overlapping
transitions, packet capacity, coupling weight or dR. Thus budgets depend on the
chosen bin grid and should be compared at fixed binning. U is an effective base
equalization rate, not measured watts; E_RF is command exposure, not calibrated
energy. Instantaneous AFP has no specified pulse energy and is excluded. The
budget does not include DNP or T1 sources.

## Using the control

1. Launch `python3 run_app.py` from `Budget` and capture the desired state in the
   tensor designer. This may be a state already manipulated by AFP.
2. Calculate with **Limit total RF exposure** unchecked to obtain a reference.
3. Click **Use result exposure as budget**, lower the number, and recalculate.
   Half the displayed reference exposure gives a 50% budget. Exact numeric
   values can be entered in **Exact / advanced settings...**.
4. Inspect Q, the no-RF comparison, exposure used, budget, and the command plot.
   Load or export the selected result using the existing workflow.

The budget is a ceiling on each generated sequence. Changes to the manual pulse
table or global gain, or replaying a sequence multiple times, change the total
delivered exposure; they are not restricted by this optimizer setting. Leaving
the limit unchecked retains the original search behavior.

Command-line examples, run from this directory:

```bash
python3 design_tensor.py --material materials/default_demo.json --rf-budget 100 --out design_100
python3 design_tensor.py --state post_afp.npz --rf-budget 50 --out design_50
python3 design_tensor.py --material saved_material.json --unlimited-rf-budget --out unrestricted
python3 design_tensor.py --replay design_50
```

`--wall-time-budget` controls search computation time; its old alias `--budget`
still has that meaning. It is independent of `--rf-budget`.

## Optimization and checks

Every trial duration enforces its own exposure constraint, so longer durations
have a smaller sum of permitted commands. For a binding cap, SLSQP optimizes
bounded normalized bin commands with a linear inequality and the existing
adjoint gradient. Per-bin bounds are tightened to at most B/T before the solve.
The initial guesses are feasible. Infeasible intermediate solver evaluations
cannot be selected as results. The final candidate is made feasible if needed
and evaluated again before selection. Unconstrained/nonbinding trials retain
the original L-BFGS-B call.

The model is never advanced by the search. Each selected endpoint is replayed
with the existing scheduler and checked at finer dt. The generated schedule's
summed exposure is checked independently against B (relative numerical allowance
1e-12). There is no change to powers after endpoint validation. The search still
selects the shortest endpoint within tolerance of its best Q found.

This remains a local numerical search over the existing pulse family. It does
not certify a global optimum or guarantee perfectly monotone gain across
independent runs. Examine convergence warnings and use consistent grid, state,
duration range and search effort for budget comparisons. Natural dynamics remain
active according to the captured flags; the objective is final Q, and the no-RF
endpoint is reported separately.

The constrained solver follows SciPy's
[SLSQP interface](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).

## Saved results and compatibility

Material JSON stores the budget in its optimizer settings. Older optimizer
settings load as unlimited; unknown fields and other missing fields are still
rejected. The AFP material/snapshot migration is retained.

`optimization_report.json` includes `integrated_base_exposure` and an
`rf_exposure_budget` object with limit, used, scheduled, remaining, fraction used,
and satisfaction status. Unlimited and zero budgets have a null fraction where
division is undefined. `duration_search.csv` also records exposure and solver for
each completed horizon. The legacy report field `budget_reached` continues to
refer to the computation-time limit, not RF expenditure.

RF JSON/CSV pulse formats are unchanged. A generated design still includes its
source snapshot, material, full report and replayable endpoint prediction.

## Tests and reproducible comparison

```bash
python3 -m pip install -r requirements-dev.txt
OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen python3 -m pytest -q
OPENBLAS_NUM_THREADS=1 python3 examples/budget_tradeoff.py --bins 701 --out outputs/my_comparison
```

The example keeps T fixed at 0.2 to isolate power reallocation, with the normal
recovery dynamics enabled. It compares unrestricted, 50%, 25% and 10% budgets
from one frozen state, writes a CSV and plot, and exports all four independent
design bundles. `--state` can supply a saved post-AFP state. These are numerical
demonstrations with uncalibrated model parameters, not measured efficiencies.

## Recorded validation (2026-10-26)

- **106 tests passed**, combining this copy's 55 AFP/persistence/budget tests
  with 51 inherited tensor-designer and Qt checks. Full output is in
  `outputs/test_results.txt`. The combined command from Budget is:
  `OPENBLAS_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen python3 -m pytest -q --import-mode=importlib ssrf_realtime/tests ../tests/test_tensor_designer.py ../tests/test_designer_gui_optional.py`.
- A direct comparison against the fixed AFP package, using negative initial P,
  an AFP-manipulated state, DNP and recovery, produced exactly equal unrestricted
  powers, selected duration and final population arrays.
- A 701-bin CLI design with budget 100 exported and replayed with maximum
  population difference **0.0**; its scheduled exposure was 99.99999999999982.
- The native offscreen Qt budget dialog was rendered and inspected at 760 × 700;
  the preview is `outputs/gui_validation/budget_designer.png`.
- The 701-bin fixed-duration demonstration in `outputs/budget_tradeoff` retained
  96.5%, 85.1% and 66.9% of the unrestricted tensor gain at 50%, 25% and 10% of
  its exposure, respectively. The constrained runs reached their 60-iteration
  local search limit; these are feasible, time-step-validated candidates, not
  certified optima. The individual reports preserve these warnings.

The inherited AFP, population dynamics, material/snapshot fixes, scheduler and
playback modules are byte-identical to the fixed AFP source copy. Only the
optimizer, designer GUI and command-line designer changed among inherited
Python modules.
