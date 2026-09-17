# Reversible spin-temperature recovery at the remaining vector polarization

## Scope and experimental condition

This revision adopts the user's experimentally established recovery condition:
with RF, DNP and ordinary T1 sources OFF, a manipulated line ultimately becomes
a Boltzmann Pake doublet at the **remaining**, not initial, vector polarization.
It changes only the internal recovery operator and its diagnostics/controls.
The RF pulse table, power conventions, event times, state grid, Pake envelope,
RF equalization, DNP target, T1 source and population-to-intensity mapping are
unchanged. No mirror peak is assigned from a hole area.

This is a thermodynamically constrained **effective kinetic model**, not a
material-specific many-body calculation. The asymptotic condition is imposed
through an ergodic set of reversible transitions, NOT by resetting or rescaling
the state to a target after a numerical step. Rates still require measured
recovery curves. The extra RF/DNP density multiplier inherited from the working
package is retained; this revision does not assert a microscopic derivation of it.

## State and observables

Each packet has n_i=(n_i+,n_i0,n_i-), sum_m n_im=mu_i, sum_i mu_i=1, and
fractional populations p_im=n_im/mu_i. The packet appears in I+ at R=x_i and
I- at R=-x_i. At a physical R the two absorption components normally belong
to different packets. The spectrum is reconstructed from these populations:

    I+(x_i)  = C_disp (n_i+ - n_i0)/DeltaR
    I-(-x_i) = C_disp (n_i0 - n_i-)/DeltaR
    P = sum_i (n_i+ - n_i-)
    Q = sum_i (n_i+ - 2n_i0 + n_i-) = 1-3 sum_i n_i0.

The total signed area is C_disp*P. An uncompensated positive-absorption RF burn
continues to remove P through the existing RF terms. None of the recovery
currents cancel that RF sink.

## 1. Retained same-branch single-quantum exchanges

For an unordered pair i<j:

    J_ij^+ = H_ij (p_0i p_+j - p_+i p_0j)
    J_ij^- = H_ij (p_-i p_0j - p_0i p_-j)

Packet i receives (J+, -J+ + J-, -J-), and packet j receives the opposite.
These exchange (+,0) with (0,+), or (0,-) with (-,0). They preserve each
global level total separately. They redistribute polarization but cannot, on
their own, equilibrate an arbitrary post-burn Q at fixed P.

    H_ij = K0 mu_i mu_j C_ij S(x_i-x_j)

The capacities count the represented spin-pair weights. C_ij is the existing
optional, symmetric spatial/orientation-correlation factor. The spectrum alone
does NOT specify real-space internuclear distances or correlations.

## 2. Cross-branch exchange is now active by default

For an ordered branch pair (+ at i, - at j), between two distinct nuclei:

    (0_i,0_j) <-> (+_i,-_j)
    J_ij^X = X_ij (p_0i p_0j - p_+i p_-j)
    X_ij = K0 r_X mu_i mu_j C_ij S(x_i+x_j).

Updates:

    dn_i+ += JX; dn_i0 -= JX
    dn_j- += JX; dn_j0 -= JX.

Each event conserves total P, but changes Q by 6 times its net event current.
The reverse population product is essential. This is not a one-way mechanism
that always increases Q: it can raise or lower Q depending on the populations.
If both nuclei belong to one packet, the code uses the existing half-count
coefficient for that unordered within-class pair. This convention changes only
that effective rate, not detailed balance; it never refers to one nucleus
making two incompatible transitions.

The resonance mismatch is x_i+x_j, not x_i-x_j, since these are opposite
transition branches. Thus the channel involves spin pairs at the *same physical
transition frequency*, not an independent mirror reservoir. The mirror signal
is still the second population difference of an already represented packet.

A connected same-branch graph makes the stationary population ratios uniform.
At stationarity, cross exchange additionally requires

    p_0^2 = p_+ p_-.

Together with normalization and fixed remaining P, this is precisely the
one-spin-temperature distribution, p_m=exp(beta*m)/(1+2 cosh beta), and

    Q_B(P) = 2-sqrt(4-3P^2).

Turning r_X back to zero deliberately restores the extra invariant N0 and no
longer models generic Boltzmann recovery. The GUI reports a warning.

## 3. Double-quantum exchange supplies missing vector-order transport

For an unordered pair of nuclei,

    (-_i,+_j) <-> (+_i,-_j)
    J_ij^DQ = D_ij (p_-i p_+j - p_+i p_-j)
    D_ij = K0 r_DQ mu_i mu_j C_ij.

Packet i receives (JDQ,0,-JDQ); j receives the negative. This conserves total P
and does not change Q, even locally. It redistributes a vector deficit without
requiring the tensor disturbance to disperse at the same rate.

This is a documented higher-order dipolar mechanism for spin-1 quadrupolar
nuclei [2], not a new direct RF transition. In the first-order high-field model,

    omega_DQ(i) = omega_+(i)+omega_-(i) = 2 omega_d,

so the first-order quadrupolar offset cancels. The channel must NOT be cut off
using the single-quantum mismatch x_i-x_j. In an isotropic spatially mixed powder
closure C_ij=1, summing the pair currents is exactly

    dn_i+|DQ = K0 r_DQ (n_i- N_+ - n_i+ N_-),
    dn_i-|DQ = -dn_i+|DQ, dn_i0|DQ=0.

N_+ and N_- here are the current evolving global totals, not external reservoirs.
This O(N) evaluation is algebraically identical to the pair sum. A nonzero
orientation-correlation setting retains the corresponding symmetric C matrix.

**Limit of this closure:** the coefficient folds in virtual-transition matrix
elements, spatial coupling strengths and their average quadrupolar denominators.
Their orientation/sign dependence is not calculated microscopically. Residual
Zeeman disorder, second-order quadrupolar shifts, and a material-specific DQ
linewidth are not yet resolved. Thus r_DQ=0.10 is a trial effective coefficient,
not a measured ND3 constant and not a claimed universal ratio of SQ/DQ rates.
The literature motivates the channel, not its numerical calibration here.

DQ exchange is not mathematically necessary to change global Q: cross exchange
does that. It is included because cross exchange alone still allowed the local
vector deficit to survive after the tensor part of a mirror feature had decayed.
The original conspicuous peak-to-hole conversion was reproduced with that
restricted transport. Adding physically allowed Zeeman-order transport changes
that transient rather than cosmetically clipping the mirror intensity.

## 4. Spectral-overlap tails and non-Zeeman energy

The default effective ZQ kernel is now Lorentzian:

    S(Delta) = 1/[1+(Delta/Gamma_ZQ)^2],

with Gamma_ZQ the HWHM in dimensionless R. This is the overlap shape associated
with an exponential correlation-time approximation. The finite-domain tails
are retained by default (cutoff=0), avoiding an artificial hard spectral boundary
that left tensor population transport extremely slow or disconnected in weak
parts of the spectrum. Frequency-offset-dependent, reservoir-assisted transport
is treated in [1,3,4]. The model does NOT claim that every measured ZQ line is
Lorentzian. Gaussian remains a selectable comparison, with width interpreted as
its standard deviation; the old calculation used Gaussian plus cutoff=4.

Cross exchange conserves nuclear Zeeman energy to leading order (fixed P), but
can exchange the small quadrupolar/dipolar mismatch with an unresolved spin bath.
The symmetric forward/reverse approximation neglects thermal factors across that
small mismatch. The resulting Boltzmann relation is the Zeeman-dominant
single-spin-temperature relation used for this experiment, not an exact Gibbs
calculation of the full quadrupolar plus dipolar Hamiltonian. Without such bath
or matching assumptions, one must track additional energies/temperatures and
cannot assume the same endpoint merely from conserving P.

## 5. Conservation, positivity and entropy

All currents separately conserve every packet's spin count and total P. Cross
exchange does not conserve N0. RF, DNP and T1 keep their separate source terms.
For each reversible reaction with forward and reverse products a,b:

    J = K(a-b),
    dS/dt = K(a-b) log(a/b) >= 0,
    S = -sum_i mu_i sum_m p_im log p_im.

Summing over reactions preserves the inequality. For a connected positive
exchange graph with cross exchange enabled, the only positive stationary state
at fixed P is the maximum-entropy Boltzmann state. The mass-action field points
inward at a zero population, hence preserves nonnegative populations in the
continuous ODE. The existing event-aware Euler integrator now includes DQ rates
in its outgoing-rate bound for safe subdivision. No negative intensity is
clipped and no conserved area is restored as a post-step correction.

## 6. DNP OFF and ON

With RF, DNP and T1 OFF, P remains P_after. All kinetics act on current
populations; the final spectrum is the analytic Pake shape with fractions
p_B(P_after). During RF, P continues changing through the RF currents. During
DNP, the unchanged target law pumps toward P_sat; cross/DQ exchange still
redistributes the evolving state. RF OFF, DNP ON and T1=0 asymptotically reaches
the same B(P_sat) target as the original DNP source. Simultaneous RF and DNP
normally give a driven non-Boltzmann state, not an enforced equilibrium line.

A Boltzmann endpoint does not prove monotonic behavior of every local intensity
for arbitrary initial states, pulse histories, and independently changed rates.
The code imposes no such clipping rule. The documented default benchmarks test
that the previous large local mirror inversion is absent, separately from the
endpoint test. Comparisons with the *pre-burn* line must allow for reduced area.

## 7. Defaults and calibration

| Setting | Default | Interpretation |
|---|---:|---|
| exchange scale K0 | 5 | Existing overall recovery time scale; uncalibrated |
| ZQ overlap | Lorentzian | Effective finite-correlation overlap model |
| zero-Q width DeltaR | 0.05 | Lorentzian HWHM; Gaussian sigma in comparison mode |
| tensor exchange ratio | 1 | Cross-branch reversible SQ exchange relative to K0 |
| DQ transport ratio | 0.10 | Effective averaged two-quantum vector transport relative to K0 |
| cutoff | 0 | Keep all overlap tails on the finite spectral grid |
| orientation correlation | 0 | No inferred correlation between EFG angle and spatial neighbors |
| MW diffusion factor | 1 | No additional microwave-induced transport change assumed |

Ratios and linewidths require joint fits of direct and mirror traces, total P,
and Q, over several RF locations and burn depths. K0 rescales time when RF and
external sources are absent. Widths and channel ratios change transient shapes
and relative modes, so changing them is not merely changing a playback speed.
The GUI shows Q-Q_B(P) and the mu-weighted RMS population distance to B(P).
These are diagnostics only: they are never used to overwrite the state.

To reproduce the old recovery exactly, set tensor exchange=0, DQ transport=0,
Gaussian overlap and cutoff=4, keeping K0 and width unchanged. This diagnostic
configuration is intentionally not a Boltzmann-equilibrating model.

## References and attribution

1. D. Suter and R. R. Ernst, *Spin diffusion in resolved solid-state NMR spectra*,
   Phys. Rev. B **32**, 5608-5627 (1985). DOI: 10.1103/PhysRevB.32.5608.
   https://doi.org/10.1103/PhysRevB.32.5608
   Discusses Zeeman/quadrupolar order, SQ/DQ channels and offset dependence.
2. R. Eckman, A. Pines, R. Tycko and D. P. Weitekamp, *Spin diffusion between
   inequivalent quadrupolar nuclei by double-quantum flip-flops*, Chem. Phys.
   Lett. **99**, 35-40 (1983). DOI: 10.1016/0009-2614(83)80265-6.
   https://doi.org/10.1016/0009-2614(83)80265-6
   Establishes a DQ channel transporting Zeeman, not quadrupolar, order even
   when single-quantum lines do not overlap. Rates depend on quadrupolar details.
3. D. Suter and R. R. Ernst, *Spectral spin diffusion in the presence of an
   extraneous dipolar reservoir*, Phys. Rev. B **25**, 6038(R) (1982).
   DOI: 10.1103/PhysRevB.25.6038.
   https://doi.org/10.1103/PhysRevB.25.6038
   Explains inverse-quadratic offset dependence with an additional dipolar bath.
4. P. A. Fedders, *Diffusion of spin order in inhomogeneous systems*, Phys. Rev.
   B **38**, 4740-4745 (1988). DOI: 10.1103/PhysRevB.38.4740.
   https://doi.org/10.1103/PhysRevB.38.4740
   Treats spin-1 order transport with additional fluctuation reservoirs.
5. D. Keller, *Selective semi-saturation measurement of spin-1*, Eur. Phys. J. A
   **62**, 36 (2026). DOI: 10.1140/epja/s10050-026-01790-y.
   https://doi.org/10.1140/epja/s10050-026-01790-y
   Provides the project's orientation-resolved recovery starting point and the
   distinction between refilling spectral features and restoring RF-lost area.

The formulas above specify the implemented reduction. They are not asserted to
be a verbatim set of equations or a fitted parameter set from any one reference.
