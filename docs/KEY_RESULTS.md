# Key results: what controls tensor enhancement

These results come from the development studies (MaterialTuning versions 1–8) that produced ssRF.
Version names map to the shipped materials as follows:

- **V5** is `d_butanol_matched`.
- **V8 default** is the V7 physical-width-exchange variant. It is not shipped, but
  `d_butanol_same_branch_transport` is it with cross-branch and DQ exchange off (study A1, ×1).
- **V1/V6** used the density-weighted RF law, which is reproducible with `rf_coupling_profile`
  (exponent 1, pedestal gain 10).
- `d_butanol_basic` is a moderate version of the horn-favouring settings in B1.

Section A used the V8 default material. For the shipped `d_butanol_matched` (V5 exchange,
mostly nonlocal DQ), the best steady RF + DNP gives Q = 0.202 instead of 0.213. The shipped
same-branch material reproduces A1 at ×1 (0.241). The study output folders are not shipped. Rerun any case with `design_tensor.py` or
`tools/steady_state_rf_dnp.py`.

All numbers come from the population-rate model, starting from a Boltzmann state at P₀ = 0.50
(Q₀ = 0.1972) unless noted. "Uncapped" means per-bin U ≤ 20 with no total power or exposure limit.
"Capped" means total U ≤ 1 (≙ 0.1 W) and exposure ≤ 34 (≙ 3.4 J).
"Continuous DNP" means the DNP target is set to P = 0.50, so DNP alone holds Q = 0.1972.
Effects of about 0.005 in Q or more are listed, plus a few smaller ones that matter for interpretation.
Status says whether an input is fitted to data or hypothetical.

## A. Continuous DNP + RF (steady operation)

| # | Mechanism | Result | Size | Status / what pins it down |
|---|---|---|---|---|
| A1 | **Type of spin-diffusion transport: same-branch vs cross-branch/DQ** | Best steady Q with the inherited mix (cross-branch = same-branch, DQ 0.084): 0.213. Same-branch only: 0.241 (K₀ fitted), 0.277 (×10), 0.301 (×100). With the mix at ×10 there is no gain. Holding the default burn fixed and strengthening the mix gives 0.179 (×10) and 0.151 (×100). | **0.088** (0.213→0.301); 0.028 at fitted K₀ | **Largest lever found. Cross-branch ratio never fitted** (inherited = 1). Equal spin-1 flip-flop matrix elements make ~1 the expected value for uncorrelated neighbours, so the same-branch-only cases need correlated orientations. Measure the refill after a narrow burn at R ≈ +1.8: does only the I− line refill, or also the I+ line at that frequency? |
| A2 | **Strength of spin diffusion (K₀)** | With same-branch transport: 0.241 → 0.277 → 0.301 for ×1, ×10, ×100. With the inherited mix, stronger diffusion *lowers* Q. | 0.06 | K₀ = 0.516 fitted to one GIF with ZQ width fixed. Exchange is weakly identified (V5: build-up insensitive to K₀ ×3). |
| A3 | **Burn placement: I−-only region** | The best steady burn goes where only the I− line exists (R ≈ +1.5 to +2.0, the I− line of the dense negative shoulder), about 0.15–0.3 U total (~15–30 mW). With same-branch transport, ~20–30% also goes onto the little peak (−0.93). | +0.015 over DNP alone (default material) | Robust across materials except horn-favouring. |
| A4 | **DNP rate** | ×10 faster DNP: best steady Q 0.2248 vs 0.2127, and burning the horn becomes worthwhile. | 0.012 | DNP rate fitted to 2014 build-up (0.00228 s⁻¹). |
| A5 | **Burning the horn at the measured DNP rate is harmful** | A continuous burn at −0.935 (U = 0.01/bin) lowers steady Q to 0.175. RF removes P faster than DNP restores it. | −0.022 | Follows from the fitted DNP rate. |
| A6 | **Pulsing / bunching RF in time** | Pulse trains (5–200× peak, 1–20% duty, 1–100 s) never beat steady. Long gaps cost up to −0.024 with fast same-branch transport (τ = 100 s); 1–5 s periods equal steady. | up to −0.024 | Incoherent saturation is concave in dose, so a steady burn is optimal. **Coherent high-power effects are not modelled.** |
| A7 | **Transient overshoot after RF switch-on** | Q exceeds its steady value a few hundred seconds after RF starts from DNP equilibrium: +0.003 (×10), +0.009 (×100 same-branch). One-shot only; a periodic schedule cannot sustain it. | 0.003–0.009 | Usable if the target is used during that window. |
| A8 | **Optimized periodic schedules / sweep direction** | 8-segment periodic schedules (τ = 30–400 s, three seeds) beat steady by ≤ +0.0016. Left→right and right→left are identical. Fixed-pattern sweeps' "motion advantages" (up to +0.014) occur only between configurations far below baseline. | ≤ 0.0016 | Robust. Timing is not a lever in this model. |
| A9 | **RF selectivity under continuous DNP** | Horn-favouring material found no steady gain (0.1972) while default and equal-rate materials gained ~0.015. | 0.015 | Local optimizer; horn-favouring suppresses shoulder coupling (θ≈0). |

## B. Single RF burn (no DNP during the burn)

| # | Mechanism | Result | Size | Status / what pins it down |
|---|---|---|---|---|
| B1 | **Per-spin RF selectivity between the two overlapping transitions** | Uncapped: horn-favouring extreme 0.2719, default (selectivity 0.7) 0.2484, equal rates 0.2372, sparse-favouring 0.2317. Capped: 0.2626 / 0.2399 / 0.2354 / 0.2246. | **0.040** | **Not identified.** The physical route is a θ/site-dependent T₂ (`rf_width_profile`). Measure narrow-hole widths and initial burn rates at horn vs pedestal, and the mirror-frequency response. |
| B2 | **RF law: V1 density-weighted vs V5 per-spin** | V1 0.2637, V5 0.2484 (uncapped); V1 law on V5 lineshape (V6) 0.2694. Code and lineshape are not the cause. | 0.015–0.021 | V1 law counts spin density twice; physically unjustified as a single-spin rate. |
| B3 | **Peak-selectivity knob (V3–V8, spectral mode)** | 0 → 0.2366, 0.7 → 0.2484, 1 → 0.2534 (uncapped V5). | 0.017 | Empirical (0.7 chosen, not fitted). |
| B4 | **Refilling strength with equal per-spin rates** | No refilling 0.2367; strong refilling concentrated at dense packets 0.2446. The optimal burn lengthens from 0.09 to ~5 time units. | 0.008 | Duration change is a measurable signature of strong refilling. |
| B5 | **RF strength calibration (U ↔ watts) under the power cap** | V7 refit strength 0.0196 → 0.0108 per U lowered capped Q 0.2441 → 0.2399. | 0.004 | Single GIF experiment; event IDs read as seconds. |
| B6 | **Power cap (0.1 W, 3.4 J)** | Costs 0.003 (V5), 0.009 (V8 default, horn-favouring), 0.002 (equal rates). | 0.002–0.009 | Hardware/calibration dependent. |
| B7 | **Lineshape (fitted multisite vs V1 single Pake)** | Under the V1 law: 0.2694 vs 0.2637. | 0.006 | Lineshape fitted (held-out RMS 0.79%). |
| B8 | **Whole-line selectivity in spectral mode** | Extending selectivity to the whole line cuts pedestal coupling ~12× and lowered capped Q (0.2401 vs 0.2453). | 0.005 | Shows coupling and selectivity must be tuned separately (V8 `rf_coupling_profile`). |
| B9 | **Time-shaped burns (4–8 segments)** | +0.0009 to +0.0025; largest with strong refilling and longer burns (0.2437 → 0.2461). | ≤ 0.0025 | Small but real; grows with refilling. |
| B10 | **Physical (narrow) ZQ exchange width (V7)** | Horns get 2.3× flip-flop partners. Fit to GIF 23% worse (RMSE 4.12e-4 vs 3.35e-4). The fit switches from nonlocal DQ (ratio 3.03) to local flip-flops (0.084). Tensor effect: none uncapped, −0.001 capped. | ~0.001 | Data mildly prefer broad/nonlocal refill. |
| B11 | **Site-resolved C–D/O–D populations and dipolar reservoir** | ≤ 0.0003 change in uncapped single-burn optima. | negligible | Matters only for slow, power-limited or thermal-mixing-dominated regimes (β·Δ up to 0.2–0.7 at capacity ratio 0.01). |

## C. Inversion (instantaneous AFP; set aside for now, recorded for completeness)

| # | Mechanism | Result | Size |
|---|---|---|---|
| C1 | Two-window AFP (R −1.05 to −0.68 and +1.15 to +2.69) | 0.298 (100% eff.), 0.278 (80%); single best window 0.250. AFP + burn up to 0.3085 (horn-favouring). | 0.10 |

## D. Open model limitations that affect interpretation

- The persistent ~20% RF-on plateau at the peak is not reproduced by any version or fit.
- Exchange parameters (K₀, width, DQ, cross-branch) are weakly identified. The cross-branch ratio has never been fitted.
- RF is incoherent (population-rate). Coherent high-power effects (nutation, spin locking, finite adiabatic passage) are absent.
- Dipolar-reservoir terms are first order in β·offset.
- The power calibration rests on one GIF experiment; event IDs are interpreted as seconds.

## Measurements in order of value

1. **Refill after a narrow burn in the I−-only region (R ≈ +1.8):** I− only, or also the I+ line at the same frequency? This pins the cross-branch ratio (A1) and K₀ (A2).
2. **Short low-power burns at horn and pedestal:** narrow-hole widths and initial fractional rates, plus the mirror-frequency response. This pins per-spin selectivity (B1) via the RF width/coupling profiles.
3. **Optimal single-burn duration versus power:** a long optimum signals strong refilling (B4).
4. **Steady RF + DNP test:** a gentle burn at R ≈ +1.9 with DNP on, watching Q rise above the DNP-only value (A3) and its ~435 s approach (A7).
