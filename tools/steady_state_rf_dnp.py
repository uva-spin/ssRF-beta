"""Best CONSTANT RF profile while DNP runs continuously (driven steady state).

    python3 tools/steady_state_rf_dnp.py d_butanol_matched --out programs/my_steady_burn.json
    python3 tools/steady_state_rf_dnp.py d_butanol_matched --dnp-target 0.5 --umax 1 --duration 3000

DNP, spin diffusion, T1 and RF are balanced exactly (Newton's method on the
population equations, reduced coordinates n+, n- per packet). The RF powers that
maximize the steady tensor polarization Q are found with L-BFGS-B using the exact
implicit-function gradient. The result is written as an RF program (every bin on
for --duration) that the GUI can load; press DNP ON, RF ON and Run to watch the
approach (several DNP time constants). A JSON report with the steady P and Q is
written next to it.

Uses the legacy (shared-site, no dipolar reservoir) dynamics; materials with
site_resolved or dipolar_enabled are rejected.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot  # noqa: E402
from ssrf_realtime.presets import material_path  # noqa: E402
from ssrf_realtime.pulse_program import BinPulse, PulseProgram, RFProfile  # noqa: E402
from ssrf_realtime.tensor_optimizer import FrozenDynamics, polarizations  # noqa: E402


class Steady:
    """Driven steady state n*(u) and dQ*/du for constant commands u."""
    def __init__(self, d):
        self.d = d; self.mu = d.mu; self.N = d.N

    def full(self, y):
        a, c = y[:self.N], y[self.N:]
        return np.c_[a, self.mu-a-c, c]

    def F(self, y, u):
        f = self.d.rhs(self.full(y), u); return np.r_[f[:, 0], f[:, 2]]

    def jac(self, y, u):
        eps = 1e-9*max(1e-6, float(self.mu.max())); f0 = self.F(y, u)
        J = np.empty((2*self.N, 2*self.N))
        for k in range(2*self.N):
            yy = y.copy(); yy[k] += eps; J[:, k] = (self.F(yy, u)-f0)/eps
        return J, f0

    def solve(self, u, y):
        for _ in range(60):
            J, f = self.jac(y, u)
            if np.max(abs(f)) < 1e-15: break
            step = np.linalg.solve(J, -f); lam = 1.0
            while lam > 1e-4:
                yn = y+lam*step
                if self.full(yn).min() > -1e-14 and np.max(abs(self.F(yn, u))) < np.max(abs(f))*(1-1e-4*lam): break
                lam *= .5
            y = yn
        J, f = self.jac(y, u)
        return y, J, float(np.max(abs(f)))

    def value_and_grad(self, u, y0):
        y, J, res = self.solve(u, y0)
        lam = np.linalg.solve(J.T, np.full(2*self.N, 3.0))          # Q = sum(3n+ + 3n-) - 2
        _, gu = self.d.vjp(self.full(y), u, np.c_[lam[:self.N], np.zeros(self.N), lam[self.N:]])
        pol = polarizations(self.full(y))
        return pol['Q'], pol['P'], -gu, y, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('material', help='preset name or material JSON')
    ap.add_argument('--out', default='programs/steady_state_burn.json', help='RF program to write')
    ap.add_argument('--dnp-target', type=float, help='override p_dnp_sat (default: material value)')
    ap.add_argument('--umax', type=float, default=1.0, help='per-bin command ceiling (U)')
    ap.add_argument('--r-limit', type=float, default=3.0, help='only bins with |R| below this may be irradiated')
    ap.add_argument('--duration', type=float, default=3000.0, help='program duration (simulation time units)')
    ap.add_argument('--maxiter', type=int, default=80)
    a = ap.parse_args()
    cfg = MaterialConfig.load(material_path(a.material))
    p = cfg.parameters
    if p['site_resolved'] or p['dipolar_enabled']:
        raise SystemExit('This tool uses the shared-site dynamics; disable site_resolved/dipolar_enabled.')
    p['dnp_enabled'] = True
    if a.dnp_target is not None: p['p_dnp_sat'] = a.dnp_target
    m = cfg.make_model(preserve_state=False)
    d = FrozenDynamics(PopulationSnapshot.capture(m)); st = Steady(d); R = m.Rplus; N = len(R)
    y0 = np.r_[d.n0[:, 0], d.n0[:, 2]]
    Q0, P0, _, ybase, _ = st.value_and_grad(np.zeros(N), y0)
    active = np.flatnonzero(abs(R) < a.r_limit); cache = {'y': ybase}
    def f(x):
        u = np.zeros(N); u[active] = x
        Q, P, g, y, _ = st.value_and_grad(u, cache['y']); cache['y'] = y
        print(f'  steady Q = {Q:.5f}  P = {P:.5f}', flush=True)
        return -1e3*Q, -1e3*g[active]
    print(f'DNP-only steady state: P = {P0:.5f}, Q = {Q0:.5f}. Optimizing RF...', flush=True)
    r = minimize(f, np.full(len(active), 1e-4), jac=True, method='L-BFGS-B',
                 bounds=[(0, a.umax)]*len(active), options=dict(maxiter=a.maxiter, ftol=1e-12))
    u = np.zeros(N); u[active] = r.x
    Q, P, _, _, res = st.value_and_grad(u, cache['y'])
    prog = PulseProgram(N, float(R[0]), float(R[-1]), profiles=[RFProfile(
        'Steady RF under continuous DNP', pulses=[BinPulse(int(j), float(u[j]), 0.0, float(a.duration))
                                                  for j in np.flatnonzero(u > 1e-9)])])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); prog.save(out)
    report = dict(material=str(material_path(a.material)), dnp_target=p['p_dnp_sat'], dnp_only=dict(P=P0, Q=Q0),
                  steady_with_rf=dict(P=P, Q=Q, residual=res), total_U=float(u.sum()),
                  irradiated_R=[float(R[j]) for j in np.flatnonzero(u > 1e-4)], program=str(out))
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('dnp_only', 'steady_with_rf', 'total_U')}, indent=2))
    print(f'Wrote {out}. Load it in the GUI with this material, turn DNP ON, then RF ON and Run.')


if __name__ == '__main__':
    main()
