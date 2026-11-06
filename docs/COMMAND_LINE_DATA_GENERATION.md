# Command line data generation

`generate_training_data.py` produces paired examples for training a machine-learning model
that reads a deuteron spectrum and outputs the RF program that maximizes the tensor
polarization Q.

- **Input:** the spectrum of a manipulated and evolved spin state.
- **Target:** the optimal RF program found by the tensor optimizer from that exact state.

It runs without the GUI, in parallel, writes sharded files and can resume after an
interruption.

## Contents
- [Quick start](#quick-start)
- [Three ways to sample states](#three-ways-to-sample-states)
- [How each sample is made](#how-each-sample-is-made)
- [Options](#options)
- [RF limits](#rf-limits)
- [Output format](#output-format)
- [Inspecting and replaying a sample](#inspecting-and-replaying-a-sample)
- [Cost and practical notes](#cost-and-practical-notes)

## Quick start

From the repository folder, after installing the requirements (see the main README):

```bash
# Random mode: 2000 samples, starting P between 20% and 60%, DNP off, 8 workers
python3 generate_training_data.py generate --out data/run1 --samples 2000 --workers 8 \
        --p0-range 20 60 --dnp off --power-cap 1 --exposure-cap 34

# Sweep mode: grid of starting P, several burns each, inputs at fixed times after the burn
python3 generate_training_data.py generate --out data/sweep1 --workers 8 --dnp on \
        --p0-values 25 27 29 31 33 35 --histories-per-p0 8 --times 0 2 5 10 20 \
        --power-cap 3 --exposure-cap 102 --post-time 30 --post-steps 3

# Chain mode (closed loop, recommended for pulse-design models): optimize, apply an
# imperfect version of the pulse, evolve 2-40 s, optimize again; 200 chains x 5 rounds
python3 generate_training_data.py generate --out data/chain1 --workers 8 --dnp on \
        --chains 200 --rounds 5 --p0-range 25 35 --power-cap 3 --exposure-cap 102 \
        --config pulse_config.json

# AFP and ssRF from the same input events, two rows each, one dataset
python3 generate_training_data.py generate --out data/both --samples 100 --ssrf --afp-profile \
        --p0-range 40 60 --dnp off --power-cap 1

python3 generate_training_data.py info --data data/sweep1            # counts, status, ranges
python3 generate_training_data.py export --data data/sweep1 --sample 17 \
        --program p17.json --state s17.npz                           # one example as GUI files
python3 run_app.py --state s17.npz --program p17.json                # open it in the GUI
```

**Resuming.** Run the same command again: shards already written are skipped. In random
mode, raising `--samples` extends a dataset; every other option must stay the same.

## Three ways to sample states

| Mode | Selected by | States | Use it for |
|---|---|---|---|
| **Chain** (closed loop) | `--chains N` | Start at P0, then repeat: optimize, apply an imperfect version of that pulse, evolve, optimize again | **The operating scenario**: states left by earlier optimized pulses, after the NMR/control loop latency |
| **Random** | `--samples N` | Each sample has its own random starting P (`--p0-range`) and random history | Broad coverage |
| **Sweep** | `--p0-values ...` | A grid of starting P values × `--histories-per-p0` random burns × `--times` after each burn | Controlled datasets; time since the burn as a model input; testing interpolation in P |

In sweep mode, all time points of one burn share the same history, so they form a true
evolution sequence. Each time point still gets its own optimized target. The total number of
samples is (number of P values) × (histories per P) × (number of times).

### Chain mode (closed loop)

ssRF does not run continuously. The realistic sequence is: measure, optimize, apply the
pulse, wait while the NMR and control loop run (a few seconds to tens of seconds), measure
again and re-optimize. Each new state needs its own pulse to bring it back to the optimum.
Chain mode generates exactly these states:

1. Round 0: a Boltzmann start at P0 (optionally after a random burn history, `--n-burns`,
   default 0–1), then one wait.
2. Optimize from the current state: this is the round's training sample.
3. Apply the program **imperfectly** (intentional undershoot), so there is always something
   left to optimize:
   - the total power is scaled by a factor from `--undershoot-power` (default 0.3–1);
   - the duration is scaled by a factor from `--undershoot-duration` (default 0.2–1);
   - the per-bin profile gets log-normal noise (`--profile-noise`, default 0.5), clipped
     to the program's own per-bin maximum;
   - with probability `--exact-prob` (default 0.15) the optimum is applied exactly.

   Undershooting only reduces power and time, so the applied pulse always respects the RF
   caps.
4. Evolve for `--round-wait` (default 2–40 s, log-uniform), with DNP as set by `--dnp`.
5. Repeat for `--rounds` optimizations.

Every optimization is a sample, so a chain of K rounds costs K optimizations and gives K
samples. `time_since_burn` is the wait since the last applied pulse (a model input),
`chain_round` the round, and `group` the chain. The applied (undershot) programs are recorded
in `events_json` as `applied_design` events.

## How each sample is made

1. **Material and start.** Pick a material from `--materials` (default `d_butanol_matched`)
   and a Boltzmann state at the chosen P.
2. **History**, run with the live simulator:
   - between `--n-burns` RF burns (sweep mode default: exactly 1), each over a random
     frequency window (`--burn-width`), with random total power (`--burn-power`) and
     duration (`--burn-duration`);
   - with probability `--afp-prob`, one AFP window;
   - random mode only: after each event, free evolution for a time from `--wait-range`, or
     none with probability `--wait-zero-prob`;
   - sweep mode only: free evolution to each of the `--times` after the burn.
3. **Input.** The total spectrum I+ + I−, which is what the NMR measures. Optionally also
   I+ and I− separately and the full level populations.
4. **Target.** Each requested manipulation runs on a copy of that state (they are not
   chained). ssRF is the tensor optimizer (per-bin U_j and duration T). `--afp-profile` is
   the GUI Q<0 AFP sweep; `final_spectrum` is stored immediately after the flip.
   `--ssrf --afp-profile` writes both rows into the same dataset, sharing `group`.
   `--samples` is the number of input events (so 100 events → 200 rows).
5. **Optional output evolution.** After the target pulse, continue for `--post-time` and
   record `--post-steps` evenly spaced spectra with their P and Q. This shows how long the
   enhancement survives.

Every sample is deterministic for a given `--seed`, so any example can be regenerated.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--out DIR` | — | Output folder |
| `--samples N` | — | Input events (random mode). With `--ssrf --afp-profile`, each event writes two rows |
| `--workers W` | physical cores − 1 | Parallel processes. SMT threads (e.g. 48 on a 24-core CPU) oversubscribe the optimizer |
| `--shard-size S` | one input event / 1 | Rows per file (random; default keeps one event together), or burn histories (sweep) or chains per file. Keep shards while generating; `generate` also writes `<dir>.npz` when it finishes |
| `--seed` | 12345 | Global seed |
| `--materials a,b` | `d_butanol_matched` | Materials to draw from (must share a frequency grid) |
| `--dnp on/off/random` | random | DNP for the whole sample. **on:** history, optimized pulse and post-pulse evolution. **off:** none. **random:** random during history waits, off during the pulse. |
| `--preset` | `fast` | Optimizer effort: `fast`, `standard` or `full` |
| `--power-cap U` | none | Total simultaneous RF power (1 = 0.1 W) |
| `--exposure-cap E` | none | Total RF energy, power × time (34 = 3.4 J) |
| `--config file.json` | — | Override any default in `ssrf_realtime/datagen.py` (`DEFAULT_CONFIG`), e.g. the optimizer duration range |
| **Random mode** | | |
| `--p0-range MIN MAX` | 0.30 0.60 | Starting P, as fraction or percent |
| `--wait-range MIN MAX` | 0 600 | Evolution after each history pulse, before the input is recorded |
| `--wait-zero-prob` | 0.2 | Chance of no evolution after a pulse |
| `--dnp-prob` | 0.3 | Chance DNP is on during each evolution (with `--dnp random`) |
| **Chain mode** | | |
| `--chains N` | — | Number of chains; enables chain mode (samples = N × rounds). P0 from `--p0-range`, or cycled over `--p0-values` |
| `--rounds K` | 4 | Optimizations per chain |
| `--round-wait MIN MAX` | 2 40 | Evolution after each applied pulse (log-uniform) |
| `--round-wait-zero-prob` | 0 | Chance of no wait |
| `--undershoot-power MIN MAX` | 0.3 1.0 | Applied / optimal total power |
| `--undershoot-duration MIN MAX` | 0.2 1.0 | Applied / optimal duration |
| `--profile-noise S` | 0.5 | Log-normal per-bin noise on the applied profile |
| `--exact-prob` | 0.15 | Chance the optimum is applied exactly |
| **Sweep mode** | | |
| `--p0-values P ...` | — | Starting P values, as percent or fractions; enables sweep mode |
| `--histories-per-p0 M` | 8 | Random burns per P value |
| `--times T ...` | 0 | Times after the burn at which inputs are taken |
| **History** | | |
| `--n-burns MIN MAX` | 0 3 (sweep: 1 1) | Number of RF burns |
| `--burn-duration MIN MAX` | 0.01 20 | Burn duration (log-uniform) |
| `--burn-power MIN MAX` | 0.01 5 | Total burn power in U (log-uniform) |
| `--burn-width MIN MAX` | 0.02 0.6 | Burn window in R (log-uniform) |
| `--afp-prob` | 0.2 (sweep: 0) | Chance of an AFP window |
| **After the target pulse** | | |
| `--post-time T`, `--post-steps K` | none | Record K spectra over time T |
| `--post-dnp` | off | DNP on during that evolution |
| `--design-dnp` | off | Optimize with DNP on during the pulse (overrides `--dnp` for that part) |
| `--no-populations` | stored | Skip the level populations to save space |
| `--states-only` | off | Skip the optimizer: store spectra with exact P, Q and branch labels only (~0.5 CPU-s per sample). For [polarization models](TRAINING_POLARIZATION_MODEL.md). |
| `--ssrf` / `--no-ssrf` | on, unless `--afp-profile` alone | Optimized RF row per input event |
| `--afp-profile` | off | Q<0 AFP row per event (`final_spectrum` right after the flip, `powers` 1 on swept bins). With `--ssrf --afp-profile`, both rows share the same history (`group`) in one file. |

Time is in simulation units, which are seconds under the d-butanol calibration. Powers are in
U, where U = 1 corresponds to 0.1 W total.

## RF limits

The targets respect the limits you give:

- the per-bin ceiling (`max_power` in the material);
- the total power, Σ_j U_j ≤ `--power-cap`;
- the total energy, T · Σ_j U_j ≤ `--exposure-cap`;
- the duration range searched.

Without caps, optimal pulses tend to put maximum power on every chosen bin for a very short
time, which is unphysical for real hardware. For realistic targets use the experimental
limits (`--power-cap 1 --exposure-cap 34`, i.e. 0.1 W and 3.4 J) or a multiple of them.

With caps, longer pulses need a matching duration range and search step. Set them through
`--config`:

```json
{"design": {"overrides": {"min_duration": 0.5, "max_duration": 60.0, "search_dt": 0.03}}}
```

## Output format

`manifest.json` holds:
- the schema;
- the full configuration and seed;
- the optimizer preset;
- the frequency grid (`grid.R`);
- the complete material files used;
- a description of every field.

Each `shard_XXXXXXXX.npz` holds arrays for its samples:

| Field | Shape | Meaning |
|---|---|---|
| `spectrum` | (n, 500) | **input:** total signal at the manipulated/evolved state |
| `time_since_burn` | (n,) | **input (sweep / chain mode):** time since the burn, or since the last applied pulse |
| `spectrum_plus`, `spectrum_minus` | (n, 500) | the two branch signals (optional) |
| `populations` | (n, 500, 3) | level populations m = +1, 0, −1 (optional) |
| `state_P`, `state_Q`, `p0` | (n,) | polarizations of the input state; starting P |
| `powers` | (n, 500) | **target:** RF command per bin |
| `duration` | (n,) | **target:** common pulse duration |
| `final_P`, `final_Q`, `delta_Q` | (n,) | predicted outcome of the target program |
| `no_rf_Q` | (n,) | Q at the same endpoint without RF (control) |
| `final_spectrum` | (n, 500) | total signal after the target program |
| `post_times`, `post_spectra`, `post_P`, `post_Q` | (n, K), (n, K, 500), … | evolution after the target pulse (optional) |
| `group` | (n,) | shared input event (dual ssRF+AFP), burn-history index (sweep), or chain index; samples sharing it are correlated |
| `chain_round` | (n,) | chain mode: optimization round (−1 otherwise) |
| `status` | (n,) | optimizer status (`completed`, `needs_finer_time_step`, …) |
| `manipulation` | (n,) | target kind: `ssrf`, `afp_profile`, or `states_only` |
| `playback_dt` | (n,) | validated integration step for replaying the target |
| `events_json` | (n,) | the history (JSON) |
| `material`, `sample_id`, `state_seconds`, `design_seconds` | (n,) | bookkeeping |

Load a dataset in Python with:

```python
from ssrf_realtime.datagen import load_dataset
data = load_dataset('data/sweep1')         # dict of arrays, sorted by sample_id
X, t, y_powers, y_T = data['spectrum'], data['time_since_burn'], data['powers'], data['duration']
```

A shard is written atomically, so it is either complete or absent. Failed samples are listed in
`failures.jsonl` with their error; the rest of the shard is kept.

## Inspecting and replaying a sample

`export` writes a sample's target as a pulse program and its input state as a population
snapshot. Both open in the GUI (`python3 run_app.py --state ... --program ...`). Replaying the
program from that state reproduces the stored `final_Q`. An exact replay needs the integration
step in `playback_dt`, which `export` prints.

## Cost and practical notes

- **Speed.** Every ssRF row runs a full optimization. With `--preset fast` this takes about
  5–10 seconds per design on one core with `--power-cap 1`, longer without a cap or with
  `--preset standard` / `full`. Use `--workers` at physical-core count, not SMT thread
  count. Run a few samples first and check `info` for the time per design.
- **Shards.** One file per input event is the default so a slow design cannot stall a
  batch of others, resume loses at most one event, and progress prints after each
  optimizer run. Do not merge shards while workers are still running. `generate` writes a
  combined `<directory>.npz` when it finishes; `combine` does the same on demand.
- **Splitting for training.** Split train/test **by `group`**. Samples of one input event
  (dual ssRF+AFP), burn history, or chain are correlated, so splitting by sample would leak
  them across sets.
- **Optimizer quality.** Targets are local optima of the same model used in the GUI. Keep only
  `status == 'completed'` for strict datasets, and weight or filter by `delta_Q` (or
  `final_Q − no_rf_Q`).
- **Noise.** No measurement noise is added. Add it during training, so one dataset serves
  several noise models.
- **Variety.** Drawing from several materials
  (`--materials d_butanol_matched,d_butanol_basic,d_butanol_extend`) keeps a model from learning
  one fixed RF selectivity.
