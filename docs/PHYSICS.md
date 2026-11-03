# Physics model (summary)

The detailed derivations are in `docs/reference/`: `PHYSICS_MODEL.md` (RF), `RECOVERY_MODEL.md`
(spin diffusion), `TENSOR_DESIGN.md` (optimizer) and `BUDGET.md` (exposure). They were written
during development and mention earlier version names.

## State and observables

The line is divided into N frequency packets on a grid R (1 R = the C–D quadrupole splitting
scale). Packet i holds a capacity μᵢ, the number of spins whose I+ transition lies at Rᵢ; this
is the powder density, including the θ Jacobian. Its I− transition lies at −Rᵢ. Level
populations are nᵢ = μᵢ·(p₊, p₀, p₋).

- P = Σ(n₊ − n₋)
- Q = Σ(n₊ − 2n₀ + n₋) = 1 − 3Σn₀
- The spectrum at R is I+(R) + I−(R). At each frequency the I+ of one packet overlaps the I−
  of its mirror packet.

With site resolution, each frequency carries separate C–D and O–D packets.

## RF (incoherent, weak drive)

A command Uⱼ is carrier power at bin j (U = 1 ≙ 0.1 W in the d-butanol calibration). Packet i
is driven at

  g₊ᵢ = cᵢ Σⱼ κΓᵢ/(Γᵢ² + (Rᵢ − Rⱼ)²) Uⱼ  and  g₋ᵢ = cᵢ Σⱼ κΓᵢ/(Γᵢ² + (−Rᵢ − Rⱼ)²) Uⱼ

with currents J₊ = g₊(n₊ − n₀) and J₋ = g₋(n₀ − n₋).

- **cᵢ** is the per-spin coupling: peak selectivity × `rf_coupling_profile`.
- **Γᵢ** is the homogeneous width (T₂) from `rf_width_profile`.

The rate per spin does not depend on how many spins share a frequency: capacity is already in μ.
Selectivity between the two overlapping transitions arises only through different c/Γ of the
two packets.

## Spin diffusion (conservative pair exchange)

Same-branch flip-flops between packets i and j proceed at K₀ μᵢμⱼ S(Rᵢ − Rⱼ)·√(fᵢfⱼ), where
S is a Lorentzian of width `zq_width_R` and f is the `exchange_profile`.

- Cross-branch exchange (an I+ of one spin with an I− of another) is scaled by
  `cross_branch_ratio`, with S(Rᵢ + Rⱼ).
- Double-quantum exchange is scaled by `double_quantum_ratio`.
- Every channel conserves packet mass and P, and is stationary at Boltzmann.
- `microwave_diffusion_factor` scales all channels while DNP is on.

## DNP and T1

Each packet relaxes toward a Boltzmann state:

- **DNP:** target P = `p_dnp_sat`, rate `dnp_rate` × density weight × `dnp_profile`.
- **T1:** target `t1_p_eq`, rate `t1_rate` × `t1_profile`.

## Dipolar reservoir (optional)

The state gains β, the inverse dipolar temperature in 1/R units, with energy E_d = −C_d β.

- Off-resonant RF absorption exchanges its energy mismatch with the reservoir:
  J = W[d − β(ω − ω_c)s].
- Flip-flops between different frequencies also exchange their mismatch with the reservoir.
  This is thermal mixing, scaled by `dipolar_exchange_coupling`.
- Detailed balance holds to first order in β·offset. Exchange conserves spin offset energy plus
  E_d exactly.
- β relaxes to the lattice value at `dipolar_relaxation_rate`.

## AFP

AFP is an instantaneous, ideal swap n₊↔n₀ at bin i and n₀↔n₋ at its mirror, with efficiency η.
Each site is swapped separately.

## Numerics and optimization

**Live simulation:** forward Euler. Steps split exactly at pulse edges and are subdivided for
stability. Significantly negative populations cause a failure rather than being clipped.

**Tensor designer:** constant power per bin over one common duration, maximizing Q at the end.
- Gradients come from the exact adjoint of the same equations; L-BFGS-B or SLSQP is used under
  the power and exposure caps.
- The duration search refines both sides of the best point.
- Every result is validated by replay and half-step checks.
- Results are local optima.

**Steady state with continuous DNP** (`tools/steady_state_rf_dnp.py`): solved with Newton's
method, with the implicit-function gradient.

## Validity limits

- **Weak, incoherent drive:** no nutation, spin locking or coherent pulse effects.
- **Dipolar terms** are first order in β·offset.
- **The cross-branch ratio, K₀ and per-spin selectivity** are weakly identified or unidentified.
- **RF and time calibration** rest on one experiment.
- **The ~20% RF-on plateau** at the peak is not reproduced.
