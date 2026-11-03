"""MaterialTuning-8 per-packet mechanism profiles: correctness checks.

Defaults equal one and are checked bit-for-bit against MaterialTuning-7 outside
the test suite (see README). Here every profile is switched on at once and the
live model, the optimizer dynamics, its adjoint and conservation are compared.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot, validate_params
from ssrf_realtime.spectral_rf import RFRateOperator
from ssrf_realtime.tensor_optimizer import FrozenDynamics

ROOT = Path(__file__).resolve().parents[1]
ALL_ON = dict(
    rf_coupling_density_exponent=.8, rf_coupling_pedestal_gain=3., rf_coupling_theta_p2=.4,
    rf_coupling_theta_p4=-.2, rf_coupling_od_factor=1.7,
    rf_width_density_exponent=.3, rf_width_theta_p2=-.3, rf_width_theta_p4=.1, rf_width_od_factor=.6,
    exchange_density_exponent=.7, exchange_theta_p2=.5, exchange_theta_p4=.2, exchange_od_factor=2.,
    dnp_theta_p2=.3, dnp_theta_p4=-.1, dnp_od_factor=1.4,
    t1_density_exponent=-.4, t1_theta_p2=.2, t1_theta_p4=.1, t1_od_factor=.5)


def material(name='target_data_butanol_v7', **changes):
    cfg = MaterialConfig.load(ROOT/f'tests/fixtures/materials/{name}.json')
    cfg.parameters.update(changes)
    return cfg


def prepared(cfg, dnp=True, t1=.003):
    cfg.parameters.update(dnp_enabled=dnp, t1_rate=t1, t1_p_eq=.05)
    m = cfg.make_model(preserve_state=False)
    rng = np.random.default_rng(3)
    f = m.n/m.mu[:, None]*np.exp(.3*rng.normal(size=m.n.shape))   # non-equilibrium state
    m.n = m.mu[:, None]*f/f.sum(axis=1, keepdims=True)
    return m


@pytest.mark.parametrize('name', ['target_data_butanol_v7', 'balanced_peak_selectivity'])
def test_profiles_are_active_positive_and_default_to_one(name):
    off = material(name).make_model(preserve_state=False)
    on = material(name, **ALL_ON).make_model(preserve_state=False)
    for prefix in ('rf_coupling', 'rf_width', 'exchange', 'dnp', 't1'):
        assert np.all(off.mechanism_profile(prefix) == 1)
        f = on.mechanism_profile(prefix)
        assert np.all(f > 0) and np.ptp(f) > .05, prefix
    assert not off.rf_rate_operator().dense and on.rf_rate_operator().dense


@pytest.mark.parametrize('name', ['target_data_butanol_v7', 'balanced_peak_selectivity'])
def test_live_derivative_matches_optimizer_rhs_with_all_profiles(name):
    m = prepared(material(name, **ALL_ON))
    u = np.random.default_rng(5).uniform(0, .05, len(m.mu))
    m.set_program(m.program) if m.program else None
    d = FrozenDynamics(PopulationSnapshot.capture(m))
    # Live derivative with the same commands: RF term from the shared operator.
    gp, gm = m.rf_rate_operator().rates(u)
    rf = np.zeros_like(m.n)
    jp = gp*(m.n[:, 0]-m.n[:, 1]); jm = gm*(m.n[:, 1]-m.n[:, 2])
    rf[:, 0] -= jp; rf[:, 1] += jp-jm; rf[:, 2] += jm
    live = m.derivative(rf_on=False, dnp_on=True)+rf
    frozen = d.rhs(m.n.copy(), u)
    assert np.max(abs(live-frozen))/np.max(abs(frozen)) < 1e-11


def test_exchange_profiles_conserve_mass_and_polarization_and_keep_boltzmann_stationary():
    m = prepared(material(**ALL_ON), dnp=False, t1=0.)
    terms = m._spin_diffusion_terms(False)
    for name, t in terms.items():
        assert np.max(abs(t.sum(axis=1))) < 1e-16, name                 # packet mass
        assert abs(np.sum(t[:, 0]-t[:, 2])) < 1e-15, name               # vector P
    eq = material(**ALL_ON).make_model(preserve_state=False)          # Boltzmann start
    for name, t in eq._spin_diffusion_terms(False).items():
        assert np.max(abs(t)) < 1e-15, name


def test_optimizer_adjoint_matches_finite_differences_with_all_profiles():
    m = prepared(material(**ALL_ON))
    d = FrozenDynamics(PopulationSnapshot.capture(m))
    rng = np.random.default_rng(7)
    n = m.n.copy(); u = rng.uniform(0, .05, d.N); adj = rng.normal(size=n.shape)
    g, gu = d.vjp(n, u, adj)
    dn = rng.normal(size=n.shape)*m.mu[:, None]; du = rng.normal(size=d.N)
    eps = 1e-6
    fd_n = (np.sum(adj*d.rhs(n+eps*dn, u))-np.sum(adj*d.rhs(n-eps*dn, u)))/(2*eps)
    fd_u = (np.sum(adj*d.rhs(n, u+eps*du))-np.sum(adj*d.rhs(n, u-eps*du)))/(2*eps)
    assert abs(fd_n-np.sum(g*dn)) < 1e-7*max(1, abs(fd_n))
    assert abs(fd_u-gu@du) < 1e-7*max(1, abs(fd_u))


def test_dense_rf_operator_with_uniform_width_equals_fft_operator():
    R = np.linspace(-3, 3, 201); c = np.random.default_rng(1).uniform(.5, 2, 201)
    fft = RFRateOperator(R, np.ones(201), strength=.03, hwhm=.02, packet_response=c)
    dense = RFRateOperator(R, np.ones(201), strength=.03, hwhm=np.full(201, .02), packet_response=c)
    u = np.random.default_rng(2).uniform(0, 1, 201); a, b = np.random.default_rng(3).normal(size=(2, 201))
    for x, y in zip(fft.rates(u), dense.rates(u)):
        assert np.allclose(x, y, rtol=1e-12, atol=1e-15)
    assert np.allclose(fft.transpose(a, b), dense.transpose(a, b), rtol=1e-11, atol=1e-14)


def test_v1_law_is_a_special_case_of_spectral_coupling():
    v6 = MaterialConfig.load(ROOT/'tests/fixtures/materials/target_data_butanol_v1rf.json').make_model()
    v8 = material(rf_peak_selectivity=0., rf_coupling_density_exponent=1.,
                  rf_coupling_pedestal_gain=10.).make_model(preserve_state=False)
    w6 = v6.rf_rate_weights(); c8 = v8.rf_packet_response()
    assert np.allclose(w6/w6.mean(), c8/c8.mean(), rtol=1e-12)


def test_wider_rf_width_lowers_on_resonance_rate_at_fixed_strength():
    base = material().make_model(preserve_state=False)
    wide = material(rf_width_od_factor=1., rf_width_theta_p2=.5).make_model(preserve_state=False)
    ratio = (wide.rf_rate_weights()/base.rf_rate_weights())*(wide.rf_hwhm_profile()/base.rf_hwhm_profile())
    assert np.allclose(ratio, 1, rtol=1e-12)


def test_invalid_profiles_are_rejected_and_old_files_migrate():
    p = asdict(material().make_model(preserve_state=False).params)
    for bad in (dict(exchange_theta_p2=-2.5), dict(dnp_od_factor=0.), dict(t1_density_exponent=9.),
                dict(rf_width_density_clip=.5), dict(rf_coupling_pedestal_gain=0.)):
        with pytest.raises(ValueError):
            validate_params({**p, **bad})
    raw = json.loads((ROOT/'tests/fixtures/materials/target_data_butanol_v7.json').read_text())
    assert 'exchange_profile' not in raw['parameters']
    assert MaterialConfig.from_dict(raw).parameters['exchange_theta_p2'] == 0.0
