> Development-era derivation reference. Version names (MaterialTuning-N) refer to the history that produced ssRF; file paths mentioned here may not exist in this distribution. See docs/PHYSICS.md for the current summary.

# Synchronized ideal tensor design: equations and numerical method

## Selected material response

MaterialTuning-2 retains AFP, lineshape, pulse scheduling, manual editing and
the exposure constraint. It adds spectral RF saturation, with legacy/empirical
compatibility modes. Recovery and DNP equations are unchanged. The new RF
model is derived and qualified in PHYSICS_MODEL.md.

For each packet k at x_k the state is (n_+, n_0, n_-) with sum mu_k. One physical
RF vector u_j acts at the two branch frequencies. In spectral mode:

    K_kj = kappa * Gamma / (Gamma^2 + (R_k - R_j)^2)
    g_plus = K u
    g_minus = reverse(K u)

In legacy/independent compatibility modes:

    g_plus,k = w_k u_k
    g_minus,k = w_k u_reverse(k)

In all modes:

    J_plus,k  = g_plus,k  (n_plus,k - n_zero,k)
    J_minus,k = g_minus,k (n_zero,k - n_minus,k)

    dn_RF,k = (-J_plus,k, J_plus,k - J_minus,k, J_minus,k)

The unchanged SQ exchange, cross-branch equilibration, DQ transport, DNP and T1
currents are added to this RF term. Current enable flags, rate constants,
capacity/coupling factors and the population state are captured together.
Neither a global Boltzmann fit nor a fixed mirror-area ratio is used.

Q(n) = sum_k(n_plus,k - 2 n_zero,k + n_minus,k) is the objective for positive
and negative P alike. The physical spectrum is signed. A negative-vector
Boltzmann line shifts unfavorable regions to the opposite side; it does not
change the definition of Q.

## Candidate selection

In physical-bin order, form population differences

    a_j = n_plus,j - n_zero,j
    b_j = (n_zero - n_minus)_reverse(j)

and the signed tensor-density proxy (a_j-b_j)/dR. The initial RF tensor-benefit
coefficient in compatibility modes is

    G_j = 3 (w_reverse(j) b_j - w_j a_j).

In spectral mode it is `G = K.T @ (3*(b-a))`; each candidate's initial gain
includes its irradiation of neighboring packets, not just the local bin.

Masks may use negative tensor density, positive G_j, their union (default), or
the full grid. Masks are optionally restricted by user-specified R intervals
and per-bin power ceilings. The mask is fixed for that design run, saved in the
export and not automatically enlarged while predicting a pulse. The full ODE
still evolves ALL population bins. A mask is a control restriction, not a claim
that an excluded bin could never be useful in a longer, more general program.

## Pulse family and objective

All active commands start together:

    u_j(t) = U_j for 0 <= t < T; 0 otherwise.

Each U_j is independently bounded by a user ceiling; a global per-bin maximum
and optional lower per-bin ceilings are supported. An optional sequence budget
adds `T * sum_j U_j <= max_total_exposure`. `null` leaves total exposure unlimited;
zero permits no RF. Each physical bin is counted once, even though its field
drives both overlapping branches. There is no dR or packet-capacity weighting
in this command-exposure sum; compare budgets on the same grid. Gain in exported
pulse programs is one. Programs replace any pre-existing program.

For each T in a logarithmic duration scan, solve the bounded problem

    maximize Q[n(T; U)] over 0 <= U_j <= U_max,j
    subject to T * sum_j U_j <= B, if a budget B is enabled.

Include U=0 as a comparator. Select the shortest completed candidate within

    max(absolute_Q_tolerance, relative_Q_tolerance * max(0, Q_best - Q_initial))

of the best fine-grid Q found. Neighboring brackets on both sides of the best
sampled candidate are refined geometrically a configurable number of times,
choosing the wider logarithmic interval first. This is not
an assertion that the best attainable Q is monotonic in T. Independent local
starts and continuation from a previous horizon are both used. The reports
identify search limits and local solver termination messages. No global
optimality or shortest-time certificate is claimed.

An optional total simultaneous power cap also enforces
`sum_j U_j <= max_total_power`, independently of the exposure cap. At each T
the effective exposure constraint is the smaller of B and T times that power
cap. Constrained solver coordinates scale with the available mean carrier
power, capped by individual ceilings. SLSQP's stopping tolerance is tied to
the requested absolute Q tolerance. A timeout retains the best feasible
evaluated iterate at that horizon instead of discarding its progress.

### Optional command regularity

`profile_smoothing_fwhm_R > 0` enables a secondary selection after the raw
duration/power optimization. Gaussian averaging constructs a candidate on the
finite grid; masked bins stay zero, and bounded scalar rescaling preserves
total power without violating any per-bin ceiling. This modifies actual RF
commands, not their plotted appearance. No smoothness penalty is added to Q.

The candidate must reduce `sum_j (U_(j+1)-U_j)^2`, pass the ordinary live-model
and half-step endpoint checks, and retain the reference RF enhancement within
`max(q_absolute_tolerance, q_relative_tolerance * (Q_raw-Q_no_RF))`.
It must also remain within the near-best duration allowance measured against
the best sampled duration, so selecting a shorter duration and a smoother
profile cannot each spend that allowance independently. The raw reference is
replayed on the same numerical grid. Q must remain at least the initial and
no-RF values. Up to six width halvings are tried if necessary; otherwise the
raw commands remain. Old saved settings default to zero width.

Exports include the unrefined command program and each refinement trial's
width, Q loss, allowed loss, and variation. The report and duration CSV also
record the maximum feasible **linearized** Q gain at the raw power solution,
computed by an exact bounded linear subproblem. This first-order gap diagnoses
local stationarity and is not a bound on global improvement. Numerical
endpoint validation and successful local solver termination remain distinct.

The no-RF endpoint at the same T is reported separately. Natural DNP or internal
equilibration may themselves increase Q; they are not attributed to RF.
When no endpoint improves on the current Q, an explicit no-pulse result is
returned instead of prescribing damaging RF. When only natural dynamics are
selected, the result is labeled `no_RF_benefit` and is not installed as a burn.

## Starting estimates and exact-vs-effective distinctions

In compatibility modes, for a single isolated noncentral bin with neither mirror irradiation nor any
other mechanism, and fixed rates, the RF-only exposure E=U*T gives

    a(E) = a(0) exp(-2 c_plus E)
    b(E) = b(0) exp(-2 c_minus E)

    Delta Q(E) = 3/2 [ b(0) (1-exp(-2 c_minus E))
                      -a(0) (1-exp(-2 c_plus E)) ].

Comparing the permitted exposure endpoints and any positive stationary exposure
gives a starting estimate, NOT the final answer. Simultaneous mirror commands
and recovery invalidate independent-bin optimization; the actual optimization
uses the complete coupled state. This calculation does not impose a mirror peak
or enforce a final local sign change.

Spectral mode instead uses the positive full-kernel initial slope for a seed.
The isolated-bin stationary-exposure formula does not apply to spectral wings.

Here w_k is model.rf_rate_weights(), including the pedestal gain in independent
mode. DNP separately uses capacity_rate_weights(). Both the old density multiplier
and the new pedestal profile are empirical response assumptions; packet capacity
is already in n. Neither is a proof of microscopic rate-density dependence. In
independent mode, rf_density_power=0 with rf_pedestal_gain=1 gives uniform RF
coupling; capacity_rate_power=0 gives uniform DNP coupling independently. Legacy
mode retains the shared RF/DNP exponent. The optimizer, adjoint, timestep bound,
seed and replay all use the same RF weights captured in the snapshot. These
empirical weights are ignored in spectral mode, which uses the shared kernel
and its transpose throughout.

## Forward and adjoint calculations

`FrozenDynamics` algebraically groups the SAME pair currents used by `model.py`.
In total-population variables the SQ and cross-branch contributions contain
bilinear products n_a,i * (K n_b)_i. For uncorrelated spatial/EFG geometry on the
uniform R grid, the spectral kernel is Toeplitz, while the cross-branch kernel
is Hankel. FFT matrix-vector products accelerate the sums without truncating
Lorentzian tails, interpolating the state or changing the kernel. The
same-packet cross-count factor 1/2 is retained. For orientation correlation or
ambiguous floating-point cutoff boundaries, exact direct matrices are used.

The forward map is explicit Euler with a fixed step for each trial horizon,
chosen below an outgoing-rate bound computed using the maximal permitted
commands for that duration: min(U_max,j, B/T) when an exposure cap is present.
The bound is independent of the candidate powers within each solve. A longer
horizon therefore does not inherit the shortest horizon's B/T ceiling. The
preflight verifies both search steps and the finer endpoint grid against
max_steps; the GUI exposes this limit and reports actionable count errors.
No significant negative population is clipped by the optimizer.
A discrete adjoint differentiates this exact time-discrete forward map:

    n_(r+1) = n_r + h F(n_r,U)
    lambda_N = grad_n Q
    lambda_r = lambda_(r+1) + h F_n(n_r,U)^T lambda_(r+1)
    grad_U Q = sum_r h F_U(n_r,U)^T lambda_(r+1).

Without a binding exposure cap, the gradient is supplied to SciPy L-BFGS-B,
preserving the original search. With a binding cap, SLSQP uses the same gradient
and a linear inequality in normalized command variables. Per-duration ceilings
also satisfy U_j <= B/T. Feasible seeds and the zero-RF control are retained;
only budget-feasible candidates can become the selected result. A final solver
point with a small constraint residual is brought inside the cap and evaluated
again before comparison. Both predicted and compiled schedule exposure are
checked before return. See BUDGET.md for the report fields and limitations.

Tests compare the RHS to the simulator with the selected response for random
non-Boltzmann populations and compare adjoint gradients with finite differences.

SciPy's official description of bounded `minimize` and analytic Jacobians:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Qt's official worker-thread guidance used for the cancellable calculation:
https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html

These implementation references concern solver/API behavior, not material-rate
calibration or proof that the selected pulse is the true physical optimum.

## Endpoint verification

Every completed horizon is reevaluated on a finer grid before selection. The
selected program is then replayed through the UNCHANGED `IdealBinModel` pulse
scheduler. The resulting endpoint is compared with a half-step calculation in
the same equations, using Q and a packet-weighted fractional-population RMS.
Since Euler is first order, the acceptance test uses twice these differences
to estimate the error of the delivered full-step endpoint. This is an
asymptotic error estimate, not a rigorous error bound. Independent adaptive
DOP853 checks cover both the experimental reference and the short, strong
demonstration pulse.
If needed, the verification halves playback dt up to the configured limit.
It never modifies RF command values or physical material rates. In MaterialTuning-3,
the GUI installs the recommended step as an internal RF-only integration limit;
it does not overwrite the material dt or slow post-RF/DNP clock advancement.
Headless replay continues to use the report's recommended dt directly.

The duration-scan CSV records each horizon's search dt and search/endpoint step
counts. The report records the selected search dt, source dt, maximum required
count, configured step limit, recommended playback dt,
verification errors and source-state fingerprint. The GUI applies a smaller
validated dt only upon explicit program installation. Loading JSON/CSV manually
later does not inherently change dt; use the reported value or the supplied
headless replay command. Source settings and the exact captured state remain
in the export for reproducibility.

## What is deliberately outside this first designer

* No Voigt RF spreading, waveform synthesis, B1-to-watts calibration or hardware actuation.
* No staggered start times, multiple time segments per bin or sequential-carrier constraint.
* No experimentally fitted material constants supplied as fact.
* No global optimum certificate or guarantee that every negative spectral contribution can be made positive.
* No change to the accepted two-plot main GUI or the meanings of the plotted intensities.
* No reconstruction of populations from an uploaded experimental trace; the input is the current simulator state.
