"""Physics evaluation: run a predicted RF program on the stored state and measure Q.

The same population model, flags and limits used to make the targets are used here,
so the optimizer's own program and a predicted one are compared on equal terms:
    gain fraction = (Q_pred - Q_noRF) / (Q_opt - Q_noRF),
with Q_noRF the no-RF evolution over the optimizer's duration (DNP/diffusion alone).
"""
import numpy as np

from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot
from ssrf_realtime.tensor_optimizer import FrozenDynamics


class PhysicsEvaluator:
    def __init__(self, manifest):
        self.manifest = manifest
        cfg = manifest['config']['design']
        self.dnp = bool(cfg['dnp_during_pulse'])
        self.cap_power = cfg['max_total_power']
        self.cap_exposure = cfg['max_total_exposure']
        ov = cfg.get('overrides', {})
        self.t_min = float(ov.get('min_duration', 0.01)); self.t_max = float(ov.get('max_duration', 60.0))
        self.dt = float(ov.get('search_dt', 0.03))
        self._dyn = {}

    def dynamics(self, sample_id, material, p0, populations):
        if sample_id not in self._dyn:
            mat = MaterialConfig.from_dict(self.manifest['materials'][str(material)])
            mat.parameters.update(p0=float(p0), q0=None)
            m = mat.make_model(preserve_state=False)
            n = np.asarray(populations, float); m.n = n*(m.mu[:, None]/n.sum(axis=1, keepdims=True))
            m.params.dnp_enabled = self.dnp; m.params.rf_enabled = False
            self._dyn[sample_id] = FrozenDynamics(PopulationSnapshot.capture(m))
        return self._dyn[sample_id]

    def legalize(self, powers, T):
        u = np.clip(np.asarray(powers, float), 0, None)
        if self.cap_power is not None and u.sum() > self.cap_power:
            u *= self.cap_power/u.sum()
        T = float(np.clip(T, self.t_min, self.t_max))
        if self.cap_exposure is not None and u.sum()*T > self.cap_exposure:
            T = self.cap_exposure/u.sum()
        return u, T

    def q(self, dyn, powers, T):
        u, T = self.legalize(powers, T)
        bound = dyn.outgoing_bound(u)
        dt = min(self.dt, 0.2/max(bound, 1e-12))
        Q, _ = dyn.evaluate(u, T, dt, max_steps=10**7)
        return Q

    def score(self, row, programs):
        """row: dict with sample fields; programs: {name: (powers, T)} -> {name: Q} plus references."""
        dyn = self.dynamics(row['sample_id'], row['material'], row['p0'], row['populations'])
        T_opt = float(row['duration'])
        out = {'optimizer': self.q(dyn, row['powers'], T_opt),
               'no_rf': self.q(dyn, np.zeros_like(row['powers']), T_opt),
               'state': float(row['state_Q'])}
        for name, (u, T) in programs.items():
            out[name] = self.q(dyn, u, T)
        return out
