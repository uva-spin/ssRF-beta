"""Unit tests for instantaneous AFP helpers and Spin1Model.afp_sweep."""
from __future__ import annotations

import numpy as np
import pytest

from ssrf_realtime.afp import (
    mirror_safe_indices,
    perform_afp_on_populations,
    qneg_sweep_indices,
    selective_window_indices,
)
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.model import PLUS, ZERO, MINUS, Spin1Model, Spin1Params
from ssrf_realtime.profile_editing import region_indices
from ssrf_realtime.pulse_program import PulseProgram, make_profile


def test_perform_afp_eta1_swaps_levels():
    n = 8
    rho_p = np.arange(n, dtype=float)
    rho_0 = np.arange(n, dtype=float) + 10.0
    rho_m = np.arange(n, dtype=float) + 20.0
    i = 2
    m = n - 1 - i
    p_before, z_before = rho_p[i], rho_0[i]
    z_m_before, m_before = rho_0[m], rho_m[m]
    perform_afp_on_populations(rho_p, rho_0, rho_m, [i], efficiency=1.0)
    assert rho_p[i] == pytest.approx(z_before)
    assert rho_0[i] == pytest.approx(p_before)
    assert rho_0[m] == pytest.approx(m_before)
    assert rho_m[m] == pytest.approx(z_m_before)


def test_mirror_safe_never_returns_both_of_a_pair():
    n = 11
    raw = list(range(n))  # includes every mirror pair
    safe = mirror_safe_indices(raw, n)
    blocked = set()
    for i in safe:
        m = n - 1 - i
        assert m not in blocked
        assert i not in blocked or i == m
        blocked.add(i)
        blocked.add(m)
    assert len(safe) == (n + 1) // 2


def test_qneg_empty_when_all_nonnegative():
    q = np.ones(20)
    assert qneg_sweep_indices(q) == []


def test_qneg_prefers_more_negative_of_mirror_pair():
    n = 10
    q = np.zeros(n)
    q[1] = -0.1
    q[n - 1 - 1] = -0.5  # mirror of 1 is more negative
    subset = qneg_sweep_indices(q)
    assert subset == [n - 1 - 1]


def test_selective_window_matches_region_after_mirror_safe():
    grid = np.linspace(-3.0, 3.0, 61)
    left, right = -1.0, -0.5
    region = list(region_indices(grid, left, right))
    expected = mirror_safe_indices(region, len(grid))
    assert selective_window_indices(grid, left, right) == expected


def test_afp_sweep_leaves_untouched_bins_identical():
    m = Spin1Model(Spin1Params(p0=0.45, n_bins=61, r_min=-3.0, r_max=3.0))
    before = m.n.copy()
    subset = [10, 12, 15]
    returned = m.afp_sweep(subset)
    n_bins = len(m.n)
    touched = set(subset) | {n_bins - 1 - i for i in subset}
    for k in range(n_bins):
        if k in touched:
            continue
        assert np.allclose(m.n[k], before[k])
    assert returned == subset
    assert m._afp_last_subset == subset


def test_afp_sweep_eta1_exchanges_plus_zero_at_subset():
    m = Spin1Model(Spin1Params(p0=0.45, n_bins=41, r_min=-3.0, r_max=3.0))
    i = 8
    before_p = float(m.n[i, PLUS])
    before_z = float(m.n[i, ZERO])
    m.params.afp_efficiency = 1.0
    m.afp_sweep([i])
    assert m.n[i, PLUS] == pytest.approx(before_z)
    assert m.n[i, ZERO] == pytest.approx(before_p)


def test_afp_then_ssrf_same_event_different_windows():
    """AFP on one L/R window, then ideal-bin RF on another, without reset."""
    m = IdealBinModel(IdealBinParams(p0=0.45, n_bins=81, r_min=-3.0, r_max=3.0, dt=0.002))
    afp_subset = selective_window_indices(m.Rplus, -1.0, -0.6)
    assert afp_subset
    P0 = m.polarizations()["P"]
    m.afp_sweep(afp_subset)
    P_after_afp = m.polarizations()["P"]
    assert abs(P_after_afp - P0) > 1e-8

    # RF program on a disjoint positive-R window; start after the AFP flip.
    program = PulseProgram(len(m.Rplus), float(m.Rplus[0]), float(m.Rplus[-1]), profiles=[])
    program.profiles.append(
        make_profile(m.Rplus, "post-AFP RF", 0.6, 1.0, shape="flat", rate=3.0, start=0.0, duration=0.05)
    )
    m.set_program(program)
    m.start_program(turn_rf_on=True)
    n_before_rf = m.n.copy()
    m.step(20, rf_on=True, dnp_on=False)
    # RF window bins should move; AFP-only bins need not stay frozen under diffusion,
    # but the RF program must have delivered exposure on its own window.
    assert float(m.total_delivered_exposure.sum()) > 0.0 or float(m.delivered_exposure.sum()) > 0.0
    assert not np.allclose(m.n, n_before_rf)
    # Event continuity: no reset — time advanced and pre-AFP snapshot still stored.
    assert m.t > 0.0
    assert m._pre_afp_state is not None
    assert m._afp_last_subset == afp_subset
