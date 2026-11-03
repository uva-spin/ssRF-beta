"""Ideal per-physical-bin RF control with the selected material response.

Only the RF input and event-aware time stepping are extended. The Pake state,
readout, RF response, spin-diffusion currents, DNP and T1 are inherited
from model.py. No Voigt function is evaluated on this path.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, Tuple
import math
import numpy as np

from .model import Spin1Model, Spin1Params
from .pulse_program import BinPulse, RFProfile, PulseProgram


@dataclass
class IdealBinParams(Spin1Params):
    # Retained base fields are zero to make the legacy comparator unambiguous.
    rf_gaussian_fwhm_R: float = 0.0
    rf_lorentzian_fwhm_R: float = 0.0


class IdealBinModel(Spin1Model):
    """One command per physical bin, shared by the two overlapping branches.

    Commands retain the old U scale (not calibrated watts). In spectral mode
    they represent carrier power, proportional to B1 squared, and drive a
    finite homogeneous response with optional empirical peak selectivity. Legacy mode
    shares capacity_rate_power/clip with DNP. Independent mode uses
    rf_density_power/clip and rf_pedestal_gain, leaving DNP weighting separate.
    """
    def __init__(self, params: Optional[Spin1Params] = None, program: Optional[PulseProgram] = None):
        super().__init__(params or IdealBinParams())
        if program is None:
            k = int(np.argmin(abs(self.Rplus-self.params.rf_burn_R)))
            program = PulseProgram(len(self.Rplus), float(self.Rplus[0]), float(self.Rplus[-1]),
                                   profiles=[RFProfile("Single-bin example", pulses=[BinPulse(k, 2.0, 0.0, 1.0)])])
        self.set_program(program)

    def reset(self) -> None:
        saved = getattr(self, "program", None)
        super().reset()
        if not np.allclose(self.Rplus, -self.Rplus[::-1], rtol=0, atol=1e-12):
            raise ValueError("Ideal profiles require a symmetric physical-R grid for exact mirror indexing")
        self._epoch = None
        self._event_times_absolute = None
        self._command_override = None
        self._stopped = False
        self.delivered_exposure = np.zeros(len(self.Rplus))
        self.total_delivered_exposure = np.zeros(len(self.Rplus))
        self.last_substeps = 0
        if saved is not None:
            saved.validate_grid(self.Rplus)
            self._compiled = saved.compile()
        self._ext_core = None; self._ext_core_key = None
        self.ext_n = None; self._ext_collapsed = None
        self.beta_d = float(self.params.dipolar_beta_initial)
        if self.extended_active():
            self._ensure_ext_state()

    # ---- MaterialTuning-8 extended engine (site-resolved / dipolar) ------
    def extended_active(self) -> bool:
        return bool(self.params.site_resolved or self.params.dipolar_enabled)

    def extended_core(self):
        """Engine for the current parameters (rebuilt when kinetics change)."""
        import json
        from dataclasses import asdict
        from .extended_core import ExtendedCore
        volatile = {'rf_enabled', 'dnp_enabled', 'diffusion_enabled', 'dt', 'rf_burn_R', 'noise_sigma',
                    'afp_subset_indices', 'afp_efficiency', 'p0', 'q0', 'dipolar_beta_initial'}
        key = json.dumps({k: v for k, v in asdict(self.params).items() if k not in volatile}, sort_keys=True)
        if self._ext_core_key != key:
            self._ext_core = ExtendedCore(self)
            self._ext_core_key = key
        return self._ext_core

    def _ensure_ext_state(self) -> None:
        """Keep the site-resolved state consistent with the public bin totals.

        If self.n was replaced externally (state edit, snapshot of totals, an
        example script), totals are redistributed over sites with equal level
        fractions per bin. AFP is applied site-wise exactly (see afp_sweep).
        """
        core = self.extended_core()
        if (self.ext_n is None or self.ext_n.shape != (core.S, len(self.mu), 3)
                or self._ext_collapsed is None or not np.array_equal(self._ext_collapsed, self.n)):
            self.ext_n = core.split(self.n)
            self._ext_collapsed = self.n.copy()

    def set_extended_state(self, ext_n, beta) -> None:
        core = self.extended_core()
        ext_n = np.array(ext_n, dtype=float)
        if ext_n.shape != (core.S, len(self.mu), 3) or not np.allclose(ext_n.sum(axis=2), core.mu, rtol=1e-10, atol=1e-15):
            raise ValueError('Extended state does not match the site capacities')
        self.ext_n = ext_n
        self.n = ext_n.sum(axis=0); self._ext_collapsed = self.n.copy()
        self.beta_d = float(beta)

    def afp_sweep(self, subset_indices=None):
        if not self.extended_active():
            return super().afp_sweep(subset_indices)
        self._ensure_ext_state()
        if subset_indices is not None:
            self.params.afp_subset_indices = [int(i) for i in subset_indices]
        self._pre_afp_state = self.n.copy()
        subset = []
        new = self.ext_n.copy()
        for s in range(new.shape[0]):
            new[s], subset = self.afp_target_state(new[s], self.params.afp_subset_indices,
                                                   efficiency=self.params.afp_efficiency)
        self.set_extended_state(new, self.beta_d)
        self._afp_last_subset = list(subset)
        return list(subset)

    def derivative(self, rf_on=None, dnp_on=None, breakdown=False):
        """Extended mode: total derivative of the bin totals (breakdown uses the shared-site legacy terms)."""
        if not self.extended_active() or breakdown:
            return super().derivative(rf_on=rf_on, dnp_on=dnp_on, breakdown=breakdown)
        rf_on = bool(self.params.rf_enabled) if rf_on is None else bool(rf_on)
        dnp_on = bool(self.params.dnp_enabled) if dnp_on is None else bool(dnp_on)
        self._ensure_ext_state()
        u = self.commanded_rf_field() if rf_on else np.zeros(len(self.mu))
        dn, _, _ = self.extended_core().rhs(self.ext_n, self.beta_d, u, dnp_on)
        return dn.sum(axis=0)

    def set_program(self, program: PulseProgram, *, integration_dt: Optional[float] = None) -> None:
        """Install a validated copy and optional RF integration-step ceiling.

        The ceiling belongs to this program, not the material time step. It
        applies only while nonzero RF is actually delivered; normal simulation
        time and RF-off recovery keep advancing at params.dt.
        """
        if integration_dt is not None:
            if isinstance(integration_dt, bool):
                raise ValueError('RF integration_dt must be positive and finite')
            integration_dt = float(integration_dt)
            if not math.isfinite(integration_dt) or integration_dt <= 0:
                raise ValueError('RF integration_dt must be positive and finite')
        candidate = program.clone()
        candidate.validate_grid(self.Rplus)
        compiled = candidate.compile()
        self.program = candidate
        self._compiled = compiled
        self.program_integration_dt = integration_dt
        self._epoch = None
        self._event_times_absolute = None
        self._command_override = None
        self._stopped = False
        self.delivered_exposure = np.zeros(len(self.Rplus))
        self.set_rf_enabled(False)

    def set_program_gain(self, gain: float) -> None:
        gain = float(gain)
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("RF gain must be finite and nonnegative")
        if not np.all(np.isfinite(gain*self._compiled.envelope)):
            raise ValueError("RF gain overflows the scheduled rates")
        self.program.gain = gain

    def start_program(self, turn_rf_on: bool = True) -> None:
        """Set program t=0 to the current simulation time, without changing n."""
        if self._compiled.end_time <= 0:
            raise ValueError("No enabled, positive-rate, positive-duration pulses are defined")
        times = self.t + self._compiled.times
        if len(times)>1 and not np.all(np.diff(times)>0):
            raise ValueError("Pulse boundaries are below floating-point time resolution; reset simulation time first")
        self._epoch = float(self.t)
        self._event_times_absolute = times
        self._stopped = False
        self.delivered_exposure.fill(0)
        if turn_rf_on:
            self.set_rf_enabled(True)

    def stop_program(self) -> None:
        self._epoch = None
        self._event_times_absolute = None
        self._stopped = True
        self.set_rf_enabled(False)

    @property
    def program_elapsed(self) -> float:
        return 0.0 if self._epoch is None else max(0.0, float(self.t-self._epoch))

    @property
    def program_state(self) -> str:
        if self._epoch is None:
            return "stopped" if self._stopped else "ready"
        if self.t >= self._event_times_absolute[-1]:
            return "finished"
        return "running" if np.any(self.commanded_rf_field()) else "waiting"

    def _interval_index(self, time: Optional[float] = None) -> int:
        if self._event_times_absolute is None:
            return -1
        return int(np.searchsorted(self._event_times_absolute, self.t if time is None else time, side="right"))-1

    def commanded_rf_field(self) -> np.ndarray:
        """Scheduled base U_j, before the master gate and packet coupling weight."""
        if self._command_override is not None:
            return self._command_override.copy()
        if self._epoch is None:
            return np.zeros(len(self.Rplus))
        return self.program.gain*self._compiled.field_for_index(self._interval_index()).copy()

    def applied_rf_field(self) -> np.ndarray:
        return self.commanded_rf_field() if self.params.rf_enabled else np.zeros(len(self.Rplus))

    def preview_rf_field(self) -> np.ndarray:
        return self.program.gain*self._compiled.envelope.copy()

    def rf_profile_physical(self, center_R=None) -> Tuple[np.ndarray, np.ndarray]:
        # The monitor location is not a source location for a whole profile.
        return self.Rplus.copy(), self.commanded_rf_field()

    def rf_profile_arrays(self, center_R=None) -> Tuple[np.ndarray, np.ndarray]:
        u = self.commanded_rf_field()
        return u, u[::-1].copy()

    def rf_rate_fields(self, center_R=None, gamma_rf=None) -> Tuple[np.ndarray, np.ndarray]:
        u = self.commanded_rf_field()
        # gamma_rf in the baseline derivative is intentionally unused: rates
        # are supplied by the per-bin program, with program.gain as the sole
        # additional global multiplier. No automatic profile normalization.
        return self.rf_rate_operator().rates(u)

    def rf_profile_summary(self, center_R=None):
        u = self.commanded_rf_field()
        return dict(profile_peak=float(np.max(u)), active_bins=int(np.count_nonzero(u)),
                    backend="ideal-bin pulse program", normalization="none")

    def effective_local_rates(self, R=None):
        if R is None: R = self.params.rf_burn_R
        kp, km = self.branch_indices(R)
        gp, gm = self.rf_rate_fields()
        u = self.commanded_rf_field()
        cap = self.local_capacity_factors(R)
        def at(v, k): return np.nan if k is None else float(v[k])
        return {**cap, "command_at_R": at(u, kp),
                "gamma_rf_Iplus_R": at(gp,kp), "gamma_rf_Iminus_R": at(gm,km),
                "gamma_rf_opposite_on_Iplus_packet": at(gm,kp),
                "gamma_rf_opposite_on_Iminus_packet": at(gp,km),
                "dnp_Iplus_R": float(self.params.dnp_rate*cap['w_Iplus_R']),
                "dnp_Iminus_R": float(self.params.dnp_rate*cap['w_Iminus_R'])}

    def rf_balance_estimate(self, R=None):
        # Avoid presenting the baseline single-carrier hold estimate for a
        # multibin schedule. This version deliberately has no optimizer.
        raise NotImplementedError("No automatic hold-rate estimation for ideal pulse programs")

    def _max_outgoing_bound(self, rf_on: bool, dnp_on: bool) -> float:
        """Conservative nonnegative-rate bound for safe Euler subdivision.

        This changes only the numerical step size when necessary, never the
        command amplitude or duration. It prevents high multibin commands from
        being silently corrected by a positivity clamp.
        """
        w = self.dnp_rate_weights()
        out = np.zeros(len(self.Rplus))
        if rf_on:
            gp, gm = self.rf_rate_fields()
            out += gp+gm
        if dnp_on:
            out += float(self.params.dnp_rate)*w
        out += self.t1_rate_profile()
        if self.params.diffusion_enabled and self.params.diffusion_scale != 0:
            c = self.diffusion_connectivity(dnp_on=dnp_on)
            out += c['same_plus0']+c['same_0minus']+c['cross_plus']+c['cross_minus']
            out += c['double_quantum']
        if not np.all(np.isfinite(out)) or np.any(out<0):
            raise ValueError("All active rates must be finite and nonnegative")
        return float(out.max())

    def step(self, n_steps: int = 1, rf_on: Optional[bool] = None, dnp_on: Optional[bool] = None) -> None:
        """Advance in simulation time, splitting exactly at every pulse edge.

        QTimer controls only redraw opportunities, not pulse durations. RF OFF
        mutes the drive while program/simulation time continues. Pause is done
        by not calling step. A completed program applies zero RF while recovery
        continues. Kinetics use the baseline forward-Euler method, with safety
        subdivision for large outgoing rates and a hard failure on nonphysical
        populations rather than clipping a significant error.
        """
        dt = float(self.params.dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if isinstance(n_steps,bool) or int(n_steps)!=n_steps or n_steps<1:
            raise ValueError("n_steps must be a positive integer")
        rf_on = bool(self.params.rf_enabled) if rf_on is None else bool(rf_on)
        dnp_on = bool(self.params.dnp_enabled) if dnp_on is None else bool(dnp_on)
        self.last_substeps = 0
        for _ in range(int(n_steps)):
            target = self.t + dt
            if target <= self.t:
                raise ValueError("dt is below floating-point resolution at the current time")
            while self.t < target:
                index = self._interval_index()
                boundary = target
                if self._event_times_absolute is not None and index+1 < len(self._event_times_absolute):
                    boundary = min(boundary, float(self._event_times_absolute[index+1]))
                self._command_override = (np.zeros(len(self.Rplus)) if self._epoch is None else
                                          self.program.gain*self._compiled.field_for_index(index).copy())
                try:
                    if self.extended_active():
                        core = self.extended_core(); self._ensure_ext_state()
                        u = self._command_override if rf_on else np.zeros(len(self.mu))
                        bound = core.outgoing_bound(u, self.beta_d, dnp_on)
                        max_h = math.inf if bound == 0 else 0.2/bound
                        if (rf_on and self.program_integration_dt is not None
                                and np.any(self._command_override > 0)):
                            max_h = min(max_h, self.program_integration_dt)
                        h = min(boundary-self.t,max_h)
                        if h <= 0 or self.t+h <= self.t:
                            raise ValueError("Unresolvable numerical substep at a pulse boundary")
                        dn, dbeta, _ = core.rhs(self.ext_n, self.beta_d, u, dnp_on)
                        candidate = self.ext_n + h*dn
                        if not np.all(np.isfinite(candidate)) or np.any(candidate < -1e-13*core.mu[:,:,None]):
                            raise FloatingPointError("Nonphysical population step; reduce dt or active rates")
                        candidate = np.maximum(candidate, 1e-30)
                        candidate *= core.mu[:,:,None]/np.maximum(candidate.sum(axis=2,keepdims=True),1e-30)
                        self.ext_n = candidate
                        self.beta_d = float(self.beta_d + h*dbeta)
                        self.n = candidate.sum(axis=0); self._ext_collapsed = self.n.copy()
                        if rf_on:
                            exposure = h*self._command_override
                            self.delivered_exposure += exposure
                            self.total_delivered_exposure += exposure
                        self.t = boundary if h == boundary-self.t else self.t+h
                        self.last_substeps += 1
                        if self.last_substeps > 200000:
                            raise RuntimeError("Too many integration substeps; lower rates or steps/tick")
                        continue
                    bound = self._max_outgoing_bound(rf_on,dnp_on)
                    max_h = math.inf if bound == 0 else 0.2/bound
                    if (rf_on and self.program_integration_dt is not None
                            and np.any(self._command_override > 0)):
                        max_h = min(max_h, self.program_integration_dt)
                    h = min(boundary-self.t,max_h)
                    if h <= 0 or self.t+h <= self.t:
                        raise ValueError("Unresolvable numerical substep at a pulse boundary")
                    dn = self.derivative(rf_on=rf_on,dnp_on=dnp_on)
                    candidate = self.n + h*dn
                    if not np.all(np.isfinite(candidate)) or np.any(candidate < -1e-13*self.mu[:,None]):
                        raise FloatingPointError("Nonphysical population step; reduce dt or active rates")
                    # Retain the baseline roundoff floor and per-packet normalization.
                    self.n = np.maximum(candidate,1e-30)
                    self.n *= self.mu[:,None]/np.maximum(self.n.sum(axis=1,keepdims=True),1e-30)
                    if rf_on:
                        exposure = h*self._command_override
                        self.delivered_exposure += exposure
                        self.total_delivered_exposure += exposure
                    self.t = boundary if h == boundary-self.t else self.t+h
                    self.last_substeps += 1
                    if self.last_substeps > 200000:
                        raise RuntimeError("Too many integration substeps; lower rates or steps/tick")
                finally:
                    self._command_override = None
