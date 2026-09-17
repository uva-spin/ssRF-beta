# Synchronized ideal tensor design: equations and numerical method

## Fixed physical model

The source is `spin1_ssrf_realtime_ideal_profiles_boltzmann.zip`. The core
`model.py`, `ideal_model.py`, lineshape, pulse scheduler, manual editor and
safe-widget implementations are preserved byte for byte. The playback correction
changes only GUI orchestration via playback.py; the model, scheduler and optimizer
remain unchanged. This feature is a
control-design and persistence layer, not another recovery-model revision.

For each packet k at x_k the state is (n_+, n_0, n_-) with sum mu_k. One physical
RF vector u_j acts at the two branch frequencies:

    g_plus,k = w_k u_k
    g_minus,k = w_k u_reverse(k)

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
coefficient is

    G_j = 3 (w_reverse(j) b_j - w_j a_j).

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
and optional lower per-bin ceilings are supported. There is no shared total RF
power budget in this ideal simultaneous-control problem. Gain in exported pulse
programs is one. Programs replace rather than add to any pre-existing program.

For each T in a logarithmic duration scan, solve the bounded problem

    maximize Q[n(T; U)] over 0 <= U_j <= U_max,j.

Include U=0 as a comparator. Select the shortest completed candidate within

    max(absolute_Q_tolerance, relative_Q_tolerance * max(0, Q_best - Q_initial))

of the best fine-grid Q found. The time bracket below the earliest near-best
candidate is refined geometrically a configurable number of times. This is not
an assertion that the best attainable Q is monotonic in T. Independent local
starts and continuation from a previous horizon are both used. The reports
identify search limits and local solver termination messages. No global
optimality or shortest-time certificate is claimed.

The no-RF endpoint at the same T is reported separately. Natural DNP or internal
equilibration may themselves increase Q; they are not attributed to RF.
When no endpoint improves on the current Q, an explicit no-pulse result is
returned instead of prescribing damaging RF. When only natural dynamics are
selected, the result is labeled `no_RF_benefit` and is not installed as a burn.

## Starting estimates and exact-vs-effective distinctions

For a single isolated noncentral bin with neither mirror irradiation nor any
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

The retained Pake-density multiplier is an empirical per-spin coupling
hypothesis inherited from the working simulator. Packet capacity is already
in n. This feature does not establish the microscopic correctness of an extra
rate-density factor. Setting the saved exponent to zero still gives the existing
uniform-coupling limit for both RF and DNP. The generator optimizes whichever
material model is explicitly selected.

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
commands. No significant negative population is clipped by the optimizer.
A discrete adjoint differentiates this exact time-discrete forward map:

    n_(r+1) = n_r + h F(n_r,U)
    lambda_N = grad_n Q
    lambda_r = lambda_(r+1) + h F_n(n_r,U)^T lambda_(r+1)
    grad_U Q = sum_r h F_U(n_r,U)^T lambda_(r+1).

The gradient is supplied to SciPy L-BFGS-B, with normalized bounded control
variables U/U_max. Tests compare the RHS to the unchanged simulator for random
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
If needed, the verification halves playback dt up to the configured limit.
It never modifies RF command values or physical material rates.

The report records the source dt, search dt, recommended playback dt,
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
