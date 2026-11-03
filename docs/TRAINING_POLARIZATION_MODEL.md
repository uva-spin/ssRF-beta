# Training a polarization model: spectrum → vector and tensor polarization

This page explains how to train a neural network that reads a deuteron NMR spectrum and outputs
the vector polarization P and the tensor polarization Q. It should work for unmanipulated
(thermal or DNP-built) signals and for signals manipulated by RF burns or AFP.

Example code: `ml/train_pq.py`. PyTorch is required
(`python3 -m pip install -r requirements-ml.txt`).

## 1. Why a model is needed

Write A₊ and A₋ for the areas of the two branches (the I₊ and I₋ transitions). Then

  **P = A₊ + A₋**, **Q = A₊ − A₋**  (normalized areas).

- **P** follows from the *total* area of the spectrum, a single calibration constant. No model
  is needed.
- **Q** needs the split into the two branches. At every frequency the I₊ line of one group of
  spins overlaps the I₋ line of another, so the total spectrum doesn't show the split directly.
  - For **unmanipulated** states, populations follow one spin temperature, and
    Q = 2 − √(4 − 3P²) (or the peak-ratio method) gives Q from P.
  - For **manipulated** states (hole burning, AFP, partial recovery), that relation is broken,
    and Q must be inferred from the *shape* of the line. That is what the network learns.

## 2. Generating the training data (fast)

Labels don't need the optimizer. Use [command line data generation](COMMAND_LINE_DATA_GENERATION.md)
with `--states-only`: it stores each state's spectrum together with its exact P and Q (and the
separate branch signals I₊, I₋). This takes about 0.5 CPU-seconds per sample instead of
minutes.

```bash
# Manipulated and unmanipulated states over a broad P range
python3 generate_training_data.py generate --out data/pq --samples 20000 --workers 8 \
    --states-only --p0-range 10 60 --n-burns 0 3 --wait-range 0 600 --afp-prob 0.3

# Extra unmanipulated states only (no burns, no AFP)
python3 generate_training_data.py generate --out data/pq_thermal --samples 5000 --workers 8 \
    --states-only --p0-range 0 70 --n-burns 0 0 --afp-prob 0
```

**Covering the right states:**
- `--n-burns 0 …` includes unmanipulated states. The evaluation separates the two kinds
  automatically, from the history recorded with each sample.
- Wider `--burn-power` / `--burn-duration` ranges and `--afp-prob` give stronger manipulations.
- `--wait-range` and `--dnp random` give partially recovered states.
- Several materials (`--materials d_butanol_matched,d_butanol_basic,d_butanol_extend`) cover
  the uncertain RF physics.
- Negative polarizations work too: `--p0-range -60 60`.

The design datasets from the [pulse-design model](TRAINING_PULSE_MODEL.md) also contain P/Q
labels for their input states and, after the optimized pulse, for `final_spectrum` and the
post-pulse spectra. They can be added as extra, strongly manipulated examples.

## 3. The model

`SpectrumToPQ` in `ml/train_pq.py`:

- **Input:** the spectrum and its mirror R → −R, divided by one **global** scale. Never
  normalize each spectrum separately: the absolute amplitude carries P.
- **Trunk:** a 1D U-Net (convolutions over the frequency axis).
- **Outputs:**
  - **P and Q**, from pooled bottleneck features;
  - **the I₊ branch signal** (auxiliary), per bin; then I₋ = total − I₊.
- **Loss:** MSE on (P, Q), plus `--branch-weight` × MSE on the I₊ branch. Supervising the
  branch split gives the network the physical structure Q depends on, instead of leaving it to
  find it from two numbers.

## 4. Training

```bash
python3 -m ml.train_pq --data data/pq --out ml_runs/pq --epochs 200 --noise 0.002
```

- `--noise` adds Gaussian noise (relative to the maximum signal) during training and testing.
  Set it to the spectrometer's noise level.
- The data are split by `group`, so states from one burn history never appear in both training
  and test sets.

## 5. Evaluation

`ml_runs/pq/report_pq.json` reports the mean absolute errors of P and Q on held-out samples, for
**all**, **unmanipulated** and **manipulated** states separately. The classical baseline is:

- **Area method:** P from the total area, with a linear calibration on the training set;
- **Boltzmann relation:** Q = 2 − √(4 − 3P²) from that P.

Expected pattern:

| | P | Q |
|---|---|---|
| Unmanipulated | The area method is essentially exact. | The Boltzmann relation is essentially exact. |
| Manipulated | The area method is still essentially exact (P = total area). | **The Boltzmann relation fails; this is where the network must win.** |

A useful model matches the baselines on unmanipulated states and clearly beats the Boltzmann
relation for Q on manipulated states. In a small smoke test, with the mild default histories,
the Boltzmann relation was off by about 0.005 in Q for manipulated states. Strong burns and AFP
push this much higher.

## 6. Applying it to measured spectra

- Put the spectrum on the training grid (`manifest.json`, `grid`). Use the same baseline
  subtraction and phase as the simulated spectra (absorptive, positive for positive P).
- Calibrate the amplitude once with a thermal-equilibrium (TE) signal of known P, so it matches
  the training scale.
- Train with noise and line-shape variations (several materials) that cover the real
  spectrometer and sample.

## 7. Limits

- The network is only as good as the simulated physics. The line shape is fitted to data, but
  the dynamics behind the manipulated states are model assumptions (see the key-results page).
- States far outside the generated histories (e.g. different RF hardware, other materials) need
  matching training data.
