# Materials and parameters

A material file (`materials/*.json`) holds every model parameter, grouped. It also stores the
tensor-designer settings (`optimizer`) and the window settings (`gui`). Load and save materials
in the GUI, or edit them with **Edit ALL material / search settings…**.

A material file is not a population state: use **Save state / Load state** for that. Older
material files load with neutral defaults for any group they lack.

## The shipped materials

### `d_butanol_matched` (default)

| Quantity | Value | Source |
|---|---|---|
| Lineshape | C–D/O–D powder mixture: C–D splitting scale 61.655 kHz, O–D/C–D ratio 1.355, η_CD 0.065, η_OD 0.033, O–D weight 2.1%, HWHM 1.45 kHz | 2014 target spectra; held-out RMS 0.79% of peak-to-peak |
| Grid | 500 bins, 1.5287 kHz spacing, R = −6.19 … +6.19 (1 R = C–D splitting) | Acquisition contract |
| DNP | rate 0.00230 s⁻¹, saturation P 0.476 | 2014 build-up; held-out RMS 0.79% |
| Spin diffusion | K₀ 0.0128, ZQ width 0.717 R, DQ/SQ 3.03, cross-branch 1 | Burn/refill of event 1406922415 (held-out RMSE 3.4×10⁻⁴); cross-branch fixed at 1 (not fitted; ~1 is expected from equal spin-1 flip-flop matrix elements) |
| RF | spectral kernel, strength 0.0196 per U, homogeneous HWHM 0.015 R; total U = 1 ≙ 0.1 W | Same burn/refill fit; event IDs read as seconds |
| RF selectivity | peak selectivity 0.7 | Empirical (V3), not identified |

### `d_butanol_basic`

The same as `d_butanol_matched` except three selectivity settings that the data do not constrain:

- `rf_peak_selectivity` = 1.0;
- `rf_coupling_density_exponent` = 0.5;
- `rf_width_theta_p2` = 0.6. Horn packets (θ = 90°) get a 2.3× narrower homogeneous RF
  linewidth than the outer shoulder (θ = 0°), which is the physically motivated route to
  selectivity: T₂ depends on orientation through dipolar couplings.

The dense:sparse per-spin rate ratio at the negative horn becomes 22:1, against 3.2:1 for the
matched material.

Check against the 0.1 W, 34 s reference burn (P₀ = 0.45), peak signal remaining:

| Material | 1 s | 3 s | 10 s | 34 s |
|---|---|---|---|---|
| Basic | 0.36 | 0.21 | 0.10 | 0.024 |
| Matched | 0.48 | 0.16 | 0.03 | 0.015 |

The reported observation is about 80% depleted after roughly 3 s. P after the burn and the
total spectrum change agree with the matched material within 0.003 and 2%. RF-off refill,
DNP build-up and the lineshape are identical by construction.

Steady RF under continuous DNP (target P = 0.50, `tools/steady_state_rf_dnp.py`):

| Material | Steady Q |
|---|---|
| Matched | 0.202 |
| Basic | 0.202 |

DNP alone gives 0.197. Most of this material's exchange is nonlocal DQ, which spreads burn
saturation onto both lines.

### `d_butanol_extend`

The same as `d_butanol_matched`, with four selectivity settings that the data do not constrain,
pushed further than in `d_butanol_basic`:

- `rf_peak_selectivity` = 1.0;
- `rf_coupling_density_exponent` = 1.0;
- `rf_width_theta_p2` = 0.6;
- `rf_width_density_exponent` = −0.3 (denser packets have a narrower RF linewidth).

The dense:sparse per-spin rate ratio at the negative horn is 83:1. Tensor designer, P₀ = 0.50,
no power cap: Q ≈ 0.275. Treat it as an upper-range setting, not a fit.

Reference burn (0.1 W, 34 s, P₀ = 0.45), peak signal remaining:

| 1 s | 3 s | 10 s | 34 s |
|---|---|---|---|
| 0.28 | 0.23 | 0.14 | 0.040 |

It matches the reported ~80% depletion in 3 s and depletes more slowly afterwards, which is a
step toward the reported persistent plateau.

### `d_butanol_full_physics`

`d_butanol_matched` plus:

- `site_resolved` = true: separate C–D and O–D populations at every frequency;
- `dipolar_enabled` = true, `dipolar_capacity_ratio` = 0.01 (an assumed order of magnitude).

### `d_butanol_same_branch_transport` (hypothesis)

Same line shape as `d_butanol_matched`, with exchange of physical (narrow) width:
`zq_width_R` = 2 × homogeneous HWHM = 0.0471 R. The exchange strength, RF strength and DNP
values were refit for that width (K₀ 0.516, RF 0.0108 per U, DNP 0.00228 s⁻¹ / 0.451).
`cross_branch_ratio` and `double_quantum_ratio` are then set to 0, so flip-flops carry
saturation only along the same line. It reproduces KEY_RESULTS A1 (×1).

It is **not a fit, and physically unlikely as it stands.** For spin 1 the cross-branch
flip-flop, which couples the I+ of one spin with the I− of a mirror-orientation spin at the
same frequency, has the same dipolar matrix element as same-branch flip-flops. A ratio near 1
is therefore expected for uncorrelated neighbours. Suppressing it requires correlated
neighbour orientations. Use this material to explore what that would mean, not as a
prediction.

### `demo_single_pake`

Single-site Pake doublet on R = −3 … 3, spectral RF, peak selectivity 0.7. Uncalibrated.

## Parameter groups

| Group | Parameters | Meaning |
|---|---|---|
| `grid` | `n_bins`, `r_min`, `r_max` | Frequency packets; symmetric grid required |
| `lineshape`, `multisite_lineshape` | `line_gamma`, `line_asym`, `line_model`, C–D/O–D splittings, η, O–D fraction | Packet capacities (powder density) |
| `initial_state` | `p0`, `q0` | Start state (Boltzmann if `q0` is null) |
| `rf_response` | `rf_response_mode` (`spectral` / `independent` / `legacy`), peak-band centre and widths | RF law |
| `rf_saturation` | `rf_spectral_strength` κ, `rf_homogeneous_hwhm_R` Γ | Rate per U: κΓ/(Γ²+ΔR²) |
| `rf_selectivity` | `rf_peak_selectivity` | Empirical density contrast inside the peak bands |
| `rf_coupling_profile` | density exponent and clip, pedestal gain, θ (P₂, P₄), O–D factor | Per-spin RF coupling across the whole line (V1 law = exponent 1, pedestal gain 10) |
| `rf_width_profile` | density exponent and clip, θ (P₂, P₄), O–D factor | Homogeneous RF width (T₂) per packet |
| `recovery` | `diffusion_scale` K₀, `zq_width_R`, `diffusion_overlap`, `cross_branch_ratio`, `double_quantum_ratio`, orientation correlation, cutoff, `microwave_diffusion_factor` | Spin diffusion (flip-flop exchange) |
| `exchange_profile` | density exponent and clip, θ, O–D factor | Per-packet exchange strength; pairs use √(fᵢfⱼ) |
| `dnp`, `dnp_profile` | `p_dnp_sat`, `dnp_rate`; θ and O–D factors | DNP pumping (with `capacity_rate_power` density weight) |
| `t1`, `t1_profile` | `t1_rate`, `t1_p_eq`; density, θ, O–D factors | Spin-lattice relaxation |
| `site_resolved` | `site_resolved`, `exchange_intersite_factor` | Separate C–D/O–D populations (multisite only) |
| `dipolar_reservoir` | `dipolar_enabled`, `dipolar_capacity_ratio`, relaxation rate, lattice and initial β, `dipolar_exchange_coupling` | Provotorov thermal mixing |
| `afp` | `afp_efficiency`, `afp_subset_indices` | Instantaneous AFP swap |
| `run_defaults`, `numerics`, `display` | monitor R, start switches, `dt`, display scaling | — |

**Profile convention:** every per-packet factor is

f = D(density) · Σ_sites w_s · m_s · [1 + a₂P₂(cosθ) + a₄P₄(cosθ)]

- D is the clipped (density/mean)^exponent, renormalized to a μ-weighted mean of 1;
- m_OD is the `*_od_factor`;
- θ is the site orientation that maps to the packet frequency (horn θ = 90°, shoulder θ = 0°).

Every default is neutral (f = 1). Validation rejects any profile that is not positive for every θ.
