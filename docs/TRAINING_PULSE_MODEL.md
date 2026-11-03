# Training a pulse-design model: spectrum → Q-optimizing RF program

This page explains how to train a neural network that reads a deuteron NMR spectrum and outputs
the RF program that maximizes the tensor polarization Q from that state. The network replaces
the tensor optimizer, which needs the full, unmeasurable level populations and minutes of
computation. Once trained, it maps a *measured* spectrum to a pulse in milliseconds.

Example code is in `ml/` (`model.py`, `data.py`, `train.py`, `physics_eval.py`). It is a
working baseline under active development, not a tuned model. PyTorch is required:

```bash
python3 -m pip install -r requirements-ml.txt
```

## 1. The learning problem

| | |
|---|---|
| **Input** | Total spectrum I(R) = I₊(R) + I₋(R) on the 500-bin grid. Optionally the time since the last burn. |
| **Output** | Per-bin RF power U_j ≥ 0 and one pulse duration T: the synchronized program the optimizer produces. |
| **Labels** | The optimizer's program, computed from the full simulated state behind each spectrum. |
| **What makes it hard** | At every frequency two different transitions overlap, so one spectrum is consistent with several population states. The network must learn which state is likely, given how the states arise. |

## 2. Generating the training data

Use [command line data generation](COMMAND_LINE_DATA_GENERATION.md). Every sample is a full
optimization (about 1–3 CPU-minutes), so plan the dataset carefully.

**Sweep mode** gives controlled datasets: a grid of starting P values × random burns × times
after each burn. Each time point gets its own optimized target:

```bash
python3 generate_training_data.py generate --out data/sweep_dnp_on --workers 8 --dnp on \
    --p0-values 25 27 29 31 33 35 --histories-per-p0 8 --times 0 2 5 10 20 \
    --power-cap 3 --exposure-cap 102 --config pulse_config.json
```

where `pulse_config.json` gives the optimizer a duration range that suits capped pulses:

```json
{"design": {"overrides": {"min_duration": 0.5, "max_duration": 60.0, "search_dt": 0.03}},
 "history": {"wait": [0.0, 60.0]}}
```

**Random mode** (`--samples N --p0-range MIN MAX`) gives broader coverage once the pipeline
works.

Recommendations:
- **Use realistic RF limits** (`--power-cap`, `--exposure-cap`). Without them the optimum puts
  maximum power on every chosen bin for a very short time, and the network only learns a bin
  mask.
- **Keep the physics settings the model will be used with**: DNP on or off, material.
  Draw from several materials if the real RF selectivity is uncertain.
- **Skip some P values** (e.g. odd percentages only) and hold out a few to test interpolation.

## 3. Preparing the data

`ml/data.py` loads the shards with `ssrf_realtime.datagen.load_dataset`. It then:

- **Filters** to `status == 'completed'` targets that use RF.
- **Splits by `group`**, not by sample. All time points of one burn are nearly the same state,
  so a per-sample split leaks near-copies into the test set. `--holdout-p0` puts whole
  P values in the test set.
- **Builds the inputs:**
  - the spectrum divided by one *global* scale (not per sample, since the absolute amplitude
    carries P);
  - its mirror image R → −R, because each frequency is coupled to its mirror;
  - a constant channel with the normalized time since the burn (sweep mode).
- **Factorizes the target:**
  - **profile** = U / ΣU, a probability distribution over bins;
  - **total power** ΣU, standardized;
  - **log T**, standardized.

## 4. The model

`ml/model.py`, `SpectrumToPulse`: a 1D U-Net over the frequency axis.

- **Encoder:** 3 down-sampling stages, convolutions with kernel 5, GroupNorm and GELU.
- **Decoder:** with skip connections; a 1×1 convolution gives per-bin profile logits.
- **Scalar head:** mean- and max-pooled bottleneck features, then an MLP giving total power and
  log T.
- **Program:** U_j = total · softmax(logits)_j, applied for T.

Before evaluation, every prediction is projected onto the RF limits (total power, total energy,
duration range), so the output is always a legal program.

## 5. Training

```bash
python3 -m ml.train --data data/sweep_dnp_on --out ml_runs/sweep --epochs 300
python3 -m ml.train --data data/sweep_dnp_on --out ml_runs/sweep_holdout --epochs 300 --holdout-p0 29 33
```

**Loss:**
- KL divergence between the target and predicted profiles, plus
- 0.5 × mean squared error on the standardized total power and log T.

AdamW with cosine learning-rate decay; the best checkpoint by validation loss is kept. It runs on
Apple GPUs (`mps`), CUDA or CPU.

## 6. Evaluating with the physics: the only metric that matters

Matching the optimizer's profile is only a proxy. `ml/physics_eval.py` takes each test sample's
stored state, runs the predicted program through the same simulator that made the labels, and
measures

  **gain fraction** = (Q_network − Q_noRF) / (Q_optimizer − Q_noRF),

where Q_noRF is the state evolved for the optimizer's duration with no RF (DNP and diffusion
alone). A value of 1 means the network matches the optimizer. Two baselines show whether the
network learned anything state-specific:

- **mean program:** the average training program, applied to every state;
- **nearest neighbour:** the program of the training sample with the closest spectrum.

`ml_runs/<name>/report.json` holds the training history, Q and gain fractions (mean, median,
10th percentile) for the network and both baselines, and per-sample results.

**Read the results against the baselines.** In narrow datasets (small P range, similar burns)
the optimal programs are similar from state to state, and the mean program can already reach
most of the optimizer's gain. A network is only useful if it clearly beats the baselines,
especially on held-out P values.

## 7. Using a trained model

The network outputs (U, T). To apply the result, run `ml.train.predict`, write the program with
`ssrf_realtime.tensor_optimizer.synchronized_program(R, U, T).save(path)`, and load it in the GUI
or on the hardware. For measured spectra:

- **Calibrate the amplitude.** Use the same global scale as in training, so the amplitude is
  consistent.
- **Add noise during training** to match the spectrometer.
- **The frequency grid must match** the dataset's grid (`manifest.json`, `grid`).

## 8. Known limits

- **Model-dependent targets.** The labels are optimal for the simulator's physics. If the real
  per-spin RF selectivity or spin-diffusion transport differs, the real optimum differs too. See
  the materials and key-results pages.
- **Local optima.** The optimizer finds local optima; filter or weight samples by `delta_Q`.
- **Synchronized programs only.** All bins on for one duration. Time-shaped programs gained
  ≤ 0.003 in Q in our studies.
- **The history distribution matters.** The network learns the states the generator produces.
  Make the histories (burn powers, widths, timings) resemble the real operating conditions.
