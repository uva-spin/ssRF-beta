"""MaterialTuning-8 extended engine: site-resolved populations and dipolar reservoir."""
import numpy as np
import pytest

from ssrf_realtime.extended_core import ExtendedCore
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot
from ssrf_realtime.tensor_optimizer import FrozenDynamics
import importlib.util as _u
from pathlib import Path as _P
_spec = _u.spec_from_file_location('_v8m', _P(__file__).with_name('test_v8_mechanisms.py'))
_v8m = _u.module_from_spec(_spec); _spec.loader.exec_module(_v8m)
ALL_ON, material = _v8m.ALL_ON, _v8m.material


def noneq(m, seed=3):
    rng = np.random.default_rng(seed)
    f = m.n/m.mu[:, None]*np.exp(.3*rng.normal(size=m.n.shape))
    return m.mu[:, None]*f/f.sum(axis=1, keepdims=True)


def ext_noneq(core, seed=4):
    rng = np.random.default_rng(seed)
    f = np.exp(.4*rng.normal(size=(core.S, core.N, 3)))*np.array([.6, .27, .13])
    return core.mu[:, :, None]*f/f.sum(axis=2, keepdims=True)


@pytest.mark.parametrize('profiles', [{}, ALL_ON])
def test_single_site_engine_reproduces_legacy_rhs_and_adjoint(profiles):
    cfg = material(**profiles); cfg.parameters.update(dnp_enabled=True, t1_rate=.002, t1_p_eq=.05)
    m = cfg.make_model(preserve_state=False); m.n = noneq(m)
    legacy = FrozenDynamics(PopulationSnapshot.capture(m))
    core = ExtendedCore(m)                         # one site, reservoir off
    u = np.random.default_rng(1).uniform(0, .05, core.N)
    dn, db, _ = core.rhs(m.n[None], 0., u, True)
    ref = legacy.rhs(m.n.copy(), u)
    assert db == 0 and np.max(abs(dn[0]-ref)) < 1e-13*np.max(abs(ref))
    adj = np.random.default_rng(2).normal(size=m.n.shape)
    g, gb, gu = core.vjp(m.n[None], 0., u, adj[None], 0., True)
    g0, gu0 = legacy.vjp(m.n.copy(), u, adj)
    assert np.allclose(g[0], g0, rtol=1e-10, atol=1e-14) and np.allclose(gu, gu0, rtol=1e-9, atol=1e-14)


def test_neutral_site_resolution_reproduces_shared_dynamics_at_equal_fractions():
    cfg = material(); cfg.parameters.update(dnp_enabled=True, site_resolved=True)
    m = cfg.make_model(preserve_state=False)
    shared = material(); shared.parameters.update(dnp_enabled=True)
    ms = shared.make_model(preserve_state=False)
    n = noneq(ms); core = m.extended_core()
    u = np.random.default_rng(1).uniform(0, .05, core.N)
    dn, _, _ = core.rhs(core.split(n), 0., u, True)
    ref = FrozenDynamics(PopulationSnapshot.capture(ms)).rhs(n.copy(), u)
    assert core.S == 2 and np.max(abs(dn.sum(axis=0)-ref)) < 1e-12*np.max(abs(ref))


def full_core(**extra):
    cfg = material(**ALL_ON)
    settings = dict(dnp_enabled=True, t1_rate=.002, t1_p_eq=.05, site_resolved=True,
                    exchange_intersite_factor=.6, dipolar_enabled=True, dipolar_capacity_ratio=.3,
                    dipolar_relaxation_rate=.01, dipolar_beta_lattice=.05)
    settings.update(extra)
    cfg.parameters.update(settings)
    m = cfg.make_model(preserve_state=False)
    return m, m.extended_core()


@pytest.mark.parametrize('beta', [0.0, 0.7])
def test_extended_adjoint_matches_finite_differences(beta):
    m, core = full_core()
    rng = np.random.default_rng(9)
    n = ext_noneq(core); u = rng.uniform(0, .05, core.N)
    a = rng.normal(size=n.shape); ab = 3.1
    g, gb, gu = core.vjp(n, beta, u, a, ab, True)
    def L(n_, b_, u_):
        dn, db, _ = core.rhs(n_, b_, u_, True); return np.sum(a*dn)+ab*db
    dn_dir = rng.normal(size=n.shape)*core.mu[:, :, None]; du = rng.normal(size=core.N); eps = 1e-6
    fd_n = (L(n+eps*dn_dir, beta, u)-L(n-eps*dn_dir, beta, u))/(2*eps)
    fd_b = (L(n, beta+eps, u)-L(n, beta-eps, u))/(2*eps)
    fd_u = (L(n, beta, u+eps*du)-L(n, beta, u-eps*du))/(2*eps)
    assert abs(fd_n-np.sum(g*dn_dir)) < 1e-7*max(1, abs(fd_n))
    assert abs(fd_b-gb) < 1e-7*max(1, abs(fd_b))
    assert abs(fd_u-gu@du) < 1e-7*max(1, abs(fd_u))


def test_exchange_conserves_mass_polarization_and_total_energy_with_reservoir():
    m, core = full_core(dnp_rate=0.0, t1_rate=0.0, dipolar_relaxation_rate=0.0)
    n = ext_noneq(core)
    for beta in (0.0, .5, -.8):
        dn, db, _ = core.rhs(n, beta, np.zeros(core.N), False)
        assert np.max(abs(dn.sum(axis=2))) < 1e-17                              # packet mass
        assert abs(np.sum(dn[..., 0]-dn[..., 2])) < 1e-16                       # vector P
        dE_spin = -np.sum(core.R[None, :]*(dn[..., 0]+dn[..., 2]))
        dE_d = -core.Cd*db
        assert abs(dE_spin+dE_d) < 1e-15*max(1, abs(dE_spin))                   # energy


def test_off_resonant_rf_cools_or_heats_the_reservoir_with_the_provotorov_sign():
    """Only one I+ transition is unsaturated; a carrier below it must cool the reservoir."""
    m, core = full_core(dnp_rate=0.0, t1_rate=0.0, dipolar_relaxation_rate=0.0, diffusion_enabled=False)
    k = int(np.argmin(abs(core.R-0.935)))
    f = np.full((core.S, core.N, 3), 1/3); f[:, k] = [.5, .25, .25]       # d+ > 0 only at packet k
    n = core.mu[:, :, None]*f
    for offset, sign in ((-3, +1), (3, -1)):       # carrier below / above that transition
        u = np.zeros(core.N); u[k+offset] = 1.
        _, db, E = core.rhs(n, 0., u, False)
        assert np.sign(db) == sign and np.sign(E) == -sign   # absorption below resonance takes reservoir energy


def test_live_extended_step_matches_optimizer_euler_and_snapshot_round_trip(tmp_path):
    cfg = material(**ALL_ON)
    cfg.parameters.update(site_resolved=True, exchange_intersite_factor=.5, dipolar_enabled=True,
                          dipolar_capacity_ratio=.5, dnp_enabled=False, dt=.01)
    live = cfg.make_model(preserve_state=False)
    live.set_extended_state(ext_noneq(live.extended_core()), .2)
    snap = PopulationSnapshot.capture(live)
    snap.save(tmp_path/'s.npz'); back = PopulationSnapshot.load(tmp_path/'s.npz').make_model()
    assert np.array_equal(back.ext_n, live.ext_n) and back.beta_d == live.beta_d
    d = FrozenDynamics(snap)
    u = np.zeros(d.N)
    _, n_frozen = d.evaluate(u, .5, .01)
    for _ in range(50): live.step(1, rf_on=False)
    assert np.max(abs(live.n-n_frozen)/live.mu[:, None]) < 1e-12


def test_site_resolution_lets_sites_diverge_and_afp_acts_per_site():
    m, core = full_core()
    m.set_extended_state(ext_noneq(core), 0.)
    before = m.ext_n.copy()
    m.afp_sweep([int(np.argmin(abs(m.Rplus+0.9)))])
    assert np.allclose(m.ext_n.sum(axis=2), core.mu) and not np.array_equal(before, m.ext_n)
    assert np.allclose(m.n, m.ext_n.sum(axis=0))
