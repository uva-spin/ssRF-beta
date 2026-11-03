> Development-era derivation reference. Version names (MaterialTuning-N) refer to the history that produced ssRF; file paths mentioned here may not exist in this distribution. See docs/PHYSICS.md for the current summary.

# MaterialTuning-5 extension: effective multisite capacities

The opt-in `multisite_butanol` mode replaces the single analytic packet density
with a C–D/O–D mixture of azimuth-averaged Dulya absorption branches. Each site
has its own splitting and eta, a common physical Lorentzian HWHM, and an inverse
splitting normalization so its fitted fraction is an integrated weight. The
mixture is normalized on the finite simulation grid to obtain packet masses.

Populations remain a `(500, 3)` array. Site contributions at the same frequency
share population fractions and kinetic parameters. This retains the existing
mirror readout, RF, spin-diffusion, DNP, AFP and optimizer equations, while
changing the physical packet capacities used by all of them. It does not
represent independent site relaxation. Nonzero orientation correlation is
rejected for this mode because a single frequency no longer identifies a
unique site orientation. Old `single_pake` materials retain the previous path.

The new preset uses fitted shape and DNP parameters. Diffusion and RF constants
remain empirical V4 priors; the build-up does not identify diffusion separately
from the DNP response. See [README.md](README.md) for the frequency mapping,
fit/holdout measurements, assumptions and source provenance. The historical
RF and recovery derivations below remain applicable to the same packet equations.

# MaterialTuning-3 extension: empirical peak selectivity

MaterialTuning-3 retains the homogeneous Lorentzian RF kernel and all recovery
operators described below. It adds an optional, state-independent packet
coefficient **c_i**. This is a tunable empirical comparison prompted by the
loss of I−/I+ crossing when Budget’s density-weighted RF law was removed. It is
not derived from the number of spins and is not a microscopic butanol model.
The earlier MaterialTuning-2 derivation below applies exactly at selectivity 0.

Let `rho_i` be the legacy power-one density response, clipped and normalized
exactly as in Budget (fixed clip 12 for this proxy), divided by its maximum.
Let `h_i` be 1 within the two peak bands and taper to 0 with the existing cubic
smoothstep over `rf_transition_width_R`. With `a = rf_peak_selectivity`:

```
c_i = 1 + h_i * (rho_i**a - 1),       0 <= a <= 1
F   = K @ U
gplus_i  = c_i * F_i
gminus_i = c_i * F_(N-1-i)
```

The new startup value is **a = 0.7**. At zero, all coefficients are exactly one.
At one, relative RF rates inside a peak band have Budget’s density contrast,
normalized to preserve the densest packet’s MaterialTuning-2 rate. Outside the
bands and their taper, c is exactly one. This coefficient can reduce weak-packet
coupling but never increases it above the MaterialTuning-2 value. No population
or observable is rescaled. Changing it affects neither RF-off recovery nor DNP.
The same coefficient multiplies both transitions of a packet; the branches at
the same physical frequency belong to different mirrored packets.

The command transpose must therefore be
`K.T @ (c * adj_plus + reverse(c * adj_minus))`.
Reversing the coefficient outside this expression would give the wrong branch
selectivity and gradient. Direct-matrix and independent finite-difference tests
cover both formulas, and optimizer candidates, stability bounds and live
playback use the same operator. RF exposure remains `T * sum(U)`.

The new material group is `rf_selectivity: {rf_peak_selectivity: 0.7}`.
Omitting the entire group migrates older material files and authenticated
snapshots to zero, preserving their old dynamics. Partial/unknown fields and
values outside [0,1] are rejected. The existing `rf_peak_center_R`,
`rf_peak_half_width_R` and `rf_transition_width_R` define the empirical bands.

A branch crossing is an output, not an enforced rule. It can reverse under
continued replenishment. The proxy, its locality to peak bands and the fixed
clip are modeling choices requiring calibration; the fundamental RF literature
cited below does not establish this density-dependent coefficient. A real fit
would require branch-resolved RF-on curves, RF-off refill and a measured B1 or
power calibration. Homogeneous linewidth and all diffusion/DNP settings remain
independently identifiable only with appropriate measurements.

---

# MaterialTuning-2: RF saturation separated from population transport

## Status and experimental target

This is a testable population-rate model, not a validated microscopic model of
deuterated butanol. The first target is the user's report of **deuterated
butanol at 1 K and 5 T**, irradiated at approximately **0.1 W for roughly 3 s**, with the
same nominal power and duration used at peaks and pedestals. The observed
ordering is easier pedestal saturation. These constraints are recorded in
`materials/deuterated_butanol_constraints.json`, separately from fitted kinetic
parameters. That file is not a loadable simulation material. The user revised
the initial 1 s order-of-magnitude estimate to closer to 3 s. At the peak the
signal initially decreases rapidly, slows substantially, and stops at roughly
20% remaining under continued RF at the same power over the observed interval.
Increasing power burns it further. The duration of that extended observation
and its detection precision have not been supplied. The pedestal reaches approximately zero quickly;
the detection threshold has not been specified. No RF-off refill time was
provided by this timing correction.

A further user comparison reports that with microwaves on, approximately
**three times more RF power** is needed to reach the same roughly 20% peak
residual. The ratio is an estimate, not a fitted rate. The corresponding
pedestal power ratio and matched initial P/Q have not been supplied.

Power and duration alone do not identify B1, homogeneous linewidth, exchange
rates or the preparation's initial polarization. The nominal supplied energy
is about 0.3 J; that is not a measurement of energy absorbed by the spin system.
The trial preset does not infer microscopic rates from field or temperature.

The new mode removes the old powder-density RF weighting and pedestal gain.
It retains the established three-level state, AFP, reversible population
exchange, separate T1/DNP, scheduler, optimizer and command-exposure budget.
Compatibility modes remain available for exact comparisons with older files.

## 1. RF coupling: derive the rate before weighting the powder

For a resolved ladder transition a↔b with off-diagonal rotating-frame
Hamiltonian element V, detuning Δω, and exponential coherence decay T2,

```
dot(c_ab) = -(1/T2 + i Δω)c_ab + i V (n_a - n_b)
J_ab      = 2 V Im(c_ab)
```

Eliminating rapidly decaying coherence gives

```
W_ab = 2 V² T2 / (1 + Δω² T2²)
J_ab = W_ab (n_a - n_b).
```

For spin 1, with H_RF = ω1 I_x and ω1 = γ B1 for the rotating RF component,
both ladder matrix elements have |V|=ω1/√2. Thus W_ab=ω1² T2/(1+Δω² T2²).
RF power scales with B1², not B1. This derivation is a weak, incoherent-drive
approximation; the population difference decays at **2W** for an isolated
two-level pair. W itself is the bidirectional transition coefficient.

Let ω_R be the angular-frequency scale per unit R and τ the seconds per
simulation-time unit. Set Γ=1/(ω_R T2), and let ω1,unit² correspond to one U.
Then the implemented kernel is

```
κ = τ ω1,unit² / ω_R
K_ij = κ Γ / (Γ² + (R_i - R_j)²)
g_plus,i  = sum_j K_ij U_j
g_minus,i = g_plus,reverse(i)
```

The two new material fields are `rf_spectral_strength` (κ) and
`rf_homogeneous_hwhm_R` (Γ). Changing Γ at fixed κ broadens the response while
lowering its on-resonance rate; it does not just broaden a peak-normalized
curve. The homogeneous RF width is distinct from the static Pake-envelope
width and from the zero-quantum exchange bandwidth.

Independent, phase-averaged carriers contribute additive powers. There is no
row normalization, edge renormalization or automatic power renormalization.
U_j is carrier power in command units, so the discrete sum has no extra dR.
When refining a distributed profile, split its power across the refined
carriers so that total power is preserved. A single carrier keeps its power.

Packet capacities μ_i already represent spin count: n_im=μ_i p_im. In RF-only
dynamics μ_i cancels from the equation for p_im. More spins increase the total
signal and total transitions, but do not by themselves slow fractional
equalization at a prescribed B1. Neither the old density exponent nor the
old protected-peak/pedestal gain enters spectral mode.

For simultaneous drives the full three-population currents remain

```
J+ = g+ (n+ - n0),  J- = g- (n0 - n-)
dot(n)|RF = (-J+, J+ - J-, J-).
```

No subtraction of an artificial spectral hole, clipping of a signal, or
rescaling of a mirror replaces these population equations. Spectra still
display the packet population differences. The static envelope is treated as
a distribution of packet center frequencies; interpreting its fitted width
as homogeneous T2 broadening as well would double-count broadening. A joint
lineshape/readout fit would be needed if homogeneous broadening is comparable
to resolved features.

## 2. Replenishment is a separate, conservative process

The existing reversible pair operator is retained as an explicit hypothesis:

```
H_ij = K0 μ_i μ_j C_ij S(R_i - R_j)          same-branch SQ
X_ij = K0 r_X μ_i μ_j C_ij S(R_i + R_j)      cross-branch SQ
D_ij = K0 r_DQ μ_i μ_j C_ij                 effective DQ
```

The cross channel uses the existing within-class pair-count convention. The
forward and reverse currents depend on the evolving fractional populations.
They conserve packet mass and total vector polarization; cross exchange can
redistribute tensor order. There is no fixed external reservoir that restores
polarization removed by RF. T1 and DNP are the separate external sources.
See `RECOVERY_MODEL.md` for the full currents and conservation identities.

K0, spectral overlap bandwidth/shape, cross/DQ weights and orientation
correlation affect **both** RF-on replenishment and RF-off refill. RF κ and Γ
do not affect recovery from an identical prepared state after RF is off.
This provides an experimental distinction absent from the old gain knob.

Suter and Ernst calculate spectral spin diffusion for quadrupolar nuclei using
dipolar coupling, frequency mismatch, bath lineshapes, and SQ/DQ transfer
channels; they also test their predictions in resolved spectra. Their work
supports these separate transport mechanisms, not universal coefficients or
a rule determined only by powder-line height. [Primary paper](https://qnap.e3.physik.tu-dortmund.de/suter/eprints/Spin_Diffusion.pdf).

**Important unresolved assumption:** a powder histogram does not specify
neighbor connectivity. C_ij is an effective closure, not a crystal geometry.
The spatially mixed default can be tested as a hypothesis for disorder, but
must not be interpreted as exchange between otherwise disconnected powder
grains. Nor does the constant DQ coefficient resolve the virtual-transition
denominators, residual DQ detunings or orientation-dependent dipolar matrix
elements. Their absence may matter for the peak/pedestal contrast. The
literature does not establish the inherited numerical values K0=5, r_X=1 and
r_DQ=0.1 for this material. No new inverse-density factor is added to hide
those limitations.

## 3. What this version can and cannot establish

The default trial uses κ=0.04, Γ=0.015 R. Its on-resonance pair rate is
2.667 per U, close to the earlier dominant-peak response, without forcing a
protected band. Neighbor irradiation and the weaker overlapping branch can
change, so complete peak trajectories are not guaranteed identical.

Independent DOP853 calculations at T=1 simulation unit give the following
commands to leave at most 1% of **each** local resonant population difference:

| Trial | Negative peak | Positive peak | Negative pedestal | Positive pedestal |
|---|---:|---:|---:|---:|
| Exchange off | 0.865 | 0.865 | 0.864 | 0.864 |
| Inherited exchange on | 4.398 | 4.285 | 3.330 | 3.318 |

The peak positions are the numerical horn positions near ±0.934 R; pedestals
are ±1.5 R. These are synthetic command units, **not watts**, and T=1 has not
been fitted to 1 experimental second. At a stricter 0.1% residual criterion,
the required commands are about 22.6–25.0 at the peaks and 24.9–25.9 at the
pedestals: the desired ordering is not robust at that depth with this trial
transport model. This is a model deficiency to investigate, not evidence that
the experimental observation is wrong. It must not be masked by picking only
the favorable saturation criterion or by introducing another pedestal boost.

A separate comparison preset, `materials/spectral_sq_dominated.json`, reduces
only the effective DQ/SQ ratio from 0.1 to 0.01. The 99.9% thresholds become
about 13.5–14.2 at the peaks and 9.6–10.0 at the pedestals, with unchanged RF
coupling and linewidth. This identifies assumed nonlocal DQ replenishment as
one contributor to the weak contrast; it does not measure that ratio or prove
it is the mechanism in butanol. The comparison's RF-off curves must also be
tested. This preset is a competing hypothesis, not the startup material.

Doubling the grid from 701 to 1401 bins changes the startup preset's reported
thresholds by at most 0.22% (`outputs/spectral_characterization/grid_convergence.json`).
The lack of a large deep-saturation contrast therefore survives this grid check.

The diagnostics compare each transition with its own initial difference and
use the worst remaining absolute fraction. Cancellation of I+ and I- is not
counted as saturation. RF-off comparisons start from the same 99%-equalized
local pairs, preserving packet mass and leaving other packets unchanged.
That controlled initial state is a diagnostic, not an exported RF pulse.

## 4. Calibration and the next level of physics

The observed fast initial drop followed by a 20% residual makes the *shape of
the time trace* a calibration target, in addition to the endpoint. Slowing
alone does not establish replenishment: even isolated RF equalization has an
exponential tail. The user's follow-up confirms persistence of the roughly
20% residual during longer irradiation at fixed power, and further burning
when power is increased. The model must therefore reproduce a power-dependent
plateau over the observed interval, rather than only a slow transient at one
chosen endpoint. A permanently RF-inaccessible 20% fraction would not explain
the additional burning at higher power; weakly driven fractions may still
produce long apparent plateaus within finite observation/detection limits.

For interpretation only, a local signal fraction s coupled to a nearly
unchanged reservoir s_b can be approximated by

```
dot(s) = -r_RF s + k_feed (s_b - s)
s_plateau / s_b = k_feed / (r_RF + k_feed).
```

Interpreting the reported 20% plateau relative to an unchanged s_b=1, this
limiting description gives k_feed/r_RF≈0.25. Here r_RF is the population-
difference decay rate (2W for an isolated pair), not W. This is a conditional
ratio, not a fitted material parameter: the observation does not give either
rate independently. It is not an extra recovery term implemented in the code.

With common RF coupling at peak and pedestal, the near-zero pedestal residual
would require much weaker relative replenishment there in this local-reservoir
limit. For example, a *hypothetical* pedestal residual below 1% would imply
k_feed,ped/r_RF < 0.0101, versus 0.25 at the peak: a feed-rate contrast greater
than about 25, under those assumptions. The user has not supplied a 1%
detection threshold, so that numerical example is not an experimental bound.
Increasing power raises r_RF and lowers this model's residual, consistent with
the reported direction of the power dependence. None of this identifies the
reservoir, establishes uniform B1, or sets absolute kinetic rates.

The simulator's internal exchange reservoir is finite and evolves. With RF
driving all packets through positive Lorentzian tails and no external sources,
the long-time state has zero polarization; it cannot support a permanently
fixed 20% residual by returning spins to their original populations. It can
show an intermediate-time plateau while remote polarization drains. With DNP
or other external replenishment active, a driven nonzero stationary signal
can be possible. The reported microwave-on/off comparison now constrains this
part of the model, while microwave status during RF-off refill remains to be
specified.

An alternative is a distribution of RF coupling/detuning with a weakly driven
fraction, giving a long tail or an unresolved residual. The current spectral
mode includes detuning across packets but assumes a common coupling strength
and homogeneous width; it does not yet resolve spatial B1 distributions within
one frequency packet. The new observation alone does not select between these
mechanisms. Also, a measured total-signal fraction must not automatically be
identified with the diagnostics' worst individual transition fraction.

### Microwave-on/off constraint: source and transport must remain distinct

The approximately threefold increase in RF power to reach the same peak
residual favors investigating enhanced replenishment while holding RF coupling
fixed as the first hypothesis. It does not uniquely establish the mechanism:
microwaves can supply nuclear polarization and may also modify its transport.
Different initial polarizations or reservoir states must be accounted for.

For the simple fixed-reservoir balance above, with identical RF conversion,
reference populations and reservoir target, the same residual at three times
the power implies approximately three times the **total effective** refill
coefficient. It would imply an additional source equivalent to twice the
microwave-off refill only if transport were unchanged and both sources had the
same target. These conditions have not been established. Therefore neither
`dnp_rate` nor `microwave_diffusion_factor` should simply be multiplied by three
on the strength of this observation. The reported ratio refers to power;
the corresponding B1 ratio would be sqrt(3) for unchanged RF-chain calibration.

Chessari et al. observed electron-polarization-dependent **proton** spin
diffusion at 1.2 K and 7 T. Their microwave irradiation reduced electron
polarization and accelerated exchange between nuclei near and farther from
paramagnetic centers. This motivates allowing a microwave-dependent transport
channel separately from DNP pumping; it neither establishes a threefold factor
nor quantitatively transfers that result to spin-1 deuterons in butanol.
[Published primary paper](https://doi.org/10.1103/PhysRevB.107.224429),
[author manuscript](https://arxiv.org/pdf/2206.14771).

The current implementation already separates these mathematical contributions:

```
dot(n) = RF[n,U] + conservative_exchange[n; K(MW)]
         + DNP_source[n; rate, target] + T1[n]
```

| Comparison hypothesis | Existing settings |
|---|---|
| Microwave off | `dnp_enabled=false`; fit the baseline exchange first |
| Added DNP pumping only | `dnp_enabled=true`, `microwave_diffusion_factor=1`, vary `dnp_rate` and the target |
| Modified transport only | `dnp_enabled=true`, `dnp_rate=0`, vary `microwave_diffusion_factor` |
| Both mechanisms | `dnp_enabled=true`, vary pumping and transport separately |

Here `dnp_enabled` also gates the existing microwave diffusion factor; the
transport-only row is a controlled model comparison, not a claim that real
microwave irradiation performs no DNP. Conservative exchange preserves total
P; the DNP source can change it. Simultaneous local and integrated-P recovery
measurements would help discriminate these effects.

The current DNP source approaches an imposed Boltzmann-like target and retains
the inherited empirical density weighting (`capacity_rate_power/clip`). That
weighting is not derived from the new microwave observation. It must be tested
against frequency-resolved buildup/refill rather than treated as a confirmed
Pake-density law. The uniform microwave diffusion factor is likewise only an
effective parameter. No kinetic settings were retuned when recording this
observation; it defines an additional constraint for a joint on/off fit.

1. Measure the initial P/Q and frequency scale, actual excitation bandwidth,
   and RF chain calibration. At the reported 0.1 W/~3 s condition, compare
   fractional residuals separately at peaks and pedestals. Also obtain several
   powers and durations; one burn curve cannot uniquely separate κ, Γ and refill.
2. Use RF-off local and mirror recovery, with microwave/DNP status recorded,
   to constrain exchange versus T1/DNP. Initial hole shape matters. Fit the
   coupled trajectories jointly, rather than assigning each bin an independent
   exponential return to its original polarization.
3. Constrain Γ using coherence/linewidth information and spectral spillover.
   Fit κ from RF-on data while using the same exchange parameters. Test holdout
   pulse widths, durations, RF budgets and signs of initial polarization.
4. If that cannot reproduce the large contrast and recovery together, resolve
   the local dipolar/EFG geometry and bath spectral densities, or add explicit
   finite reservoirs with conservative exchanges. Do not identify local
   neighbor density with the powder density without microstructural evidence.

For strong coherent irradiation, short pulses on the T2 scale, spin locking,
correlated carrier phases or non-Zeeman reservoir heating, extend to a spin-1
density-matrix treatment with explicit relaxation/exchange and, where needed,
dipolar-reservoir dynamics. The present command-only scheduler cannot infer
carrier phases. Vugmeyster, Ostrovsky and Fu use full Liouvillian evolution
for static deuteron Q-CEST; this illustrates why a population closure is not
universal. Their molecular exchange is not evidence for a specific spin-
diffusion mechanism in butanol. [Primary paper](https://doi.org/10.1002/cphc.201901053).
Redfield's solid-state saturation work likewise distinguishes the strong-field
regime from a simple Bloch treatment. [Primary paper](https://doi.org/10.1103/PhysRev.98.1787).

Until physical frequency/time/B1 scales are supplied, the weak-drive condition
ω1 T2 ≪ 1 cannot be checked from U alone. This version is therefore a bounded
model hypothesis with limiting-case validation, not a claim that the reported
0.1 W irradiation is in that regime.

## 5. Optimizer, budget and reproducibility

Simulation and optimizer share the same linear RF operator. The adjoint uses
its transpose, and the initial candidate slope includes irradiation of all
neighboring packets. Stability limits include the sum of broadened rates.
FFT multiplication implements the finite Toeplitz kernel without periodic
wraparound; direct-matrix comparisons check it independently.

The retained budget is ∫sum_j U_j dt, including each physical carrier once.
Homogeneous broadening changes the affected spins, not the supplied power
count. With measured P_unit watts per U and τ seconds per time unit, nominal
energy is P_unit τ times this exposure. Instantaneous AFP still has no modeled
RF energy cost. Existing exports retain their file schemas and old materials
remain in their original response modes; missing new fields get inactive
defaults in those modes. Snapshot fingerprints are authenticated before
migration. New spectral parameters participate in snapshot staleness checks.

Run `python3 examples/characterize_saturation.py` for independent burn/refill
curves and thresholds, or add `--bins 1401` and a new `--out` directory to check
grid resolution. The Lorentzian is sampled at packet centers; narrow features
require grid convergence. The tests also compare the rate limit to exact
isolated-transition coherence evolution, verify finite-difference adjoints,
mass conservation, material migration and budgeted post-AFP replay.

The complete 701-bin search was also run with a 600-second computation allowance:
all seven durations and three refinements finished in about 348 seconds.
Q changed from 0.15812595 to 0.19967233 at T=1.6038131. The half-step Q difference
was 4.33e-6, below the 2e-5 validation tolerance. Some local solves reached their
iteration limits; this is a numerically checked candidate, not a global optimum.
The unchanged 300-second default time allowance can return an earlier candidate.
Its partial-search status/warnings remain visible. Outputs are in
`outputs/spectral_complete_design`; source-preservation and test evidence are
in `outputs/spectral_validation`.
