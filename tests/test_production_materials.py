"""Shipped materials, launcher name resolution and the steady-state RF+DNP tool."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fixture_materials import fixture_material
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot
from ssrf_realtime.presets import available, default_material, material_path
from ssrf_realtime.pulse_program import PulseProgram
from ssrf_realtime.tensor_optimizer import FrozenDynamics

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ['d_butanol_basic', 'd_butanol_extend', 'd_butanol_matched', 'd_butanol_full_physics',
           'd_butanol_same_branch_transport', 'demo_single_pake']
FITTED = ['n_bins', 'r_min', 'r_max', 'line_gamma', 'line_asym', 'line_model', 'line_cd_scale_R', 'line_od_scale_ratio',
          'line_eta_cd', 'line_eta_od', 'line_od_fraction', 'rf_spectral_strength', 'rf_homogeneous_hwhm_R',
          'diffusion_scale', 'zq_width_R', 'cross_branch_ratio', 'double_quantum_ratio', 'dnp_rate', 'p_dnp_sat']


def test_shipped_set_and_default():
    assert available() == sorted(SHIPPED)
    assert default_material().name == MaterialConfig.load(material_path('d_butanol_matched')).name


@pytest.mark.parametrize('name', SHIPPED)
def test_each_material_loads_and_steps(name):
    m = MaterialConfig.load(material_path(name)).make_model(preserve_state=False)
    assert m.polarizations()['P'] == pytest.approx(0.5)
    m.params.dnp_enabled = True
    m.step(5)
    assert np.allclose(m.n.sum(axis=1), m.mu)


def test_experimental_material_keeps_every_fitted_value():
    fit = fixture_material('target_data_butanol').parameters
    exp = MaterialConfig.load(material_path('d_butanol_matched')).parameters
    for key in FITTED:
        assert exp[key] == fit[key], key


def test_enhanced_differs_only_in_unconstrained_rf_selectivity():
    exp = MaterialConfig.load(material_path('d_butanol_matched')).parameters
    enh = MaterialConfig.load(material_path('d_butanol_basic')).parameters
    changed = {k for k in exp if exp[k] != enh[k]}
    assert changed == {'rf_peak_selectivity', 'rf_coupling_density_exponent', 'rf_width_theta_p2'}


def test_extend_differs_only_in_unconstrained_rf_selectivity():
    exp = MaterialConfig.load(material_path('d_butanol_matched')).parameters
    ext = MaterialConfig.load(material_path('d_butanol_extend')).parameters
    changed = {k for k in exp if exp[k] != ext[k]}
    assert changed == {'rf_peak_selectivity', 'rf_coupling_density_exponent', 'rf_width_theta_p2', 'rf_width_density_exponent'}


def test_reference_program_matches_the_d_butanol_grid():
    program = PulseProgram.load(ROOT/'programs/reference_0p1W_burn.json')
    m = MaterialConfig.load(material_path('d_butanol_matched')).make_model(preserve_state=False)
    m.set_program(program)
    assert sum(p.rate for prof in program.profiles for p in prof.pulses) == pytest.approx(1.0)   # total U = 0.1 W


def test_steady_state_tool_solver_reproduces_dnp_only_equilibrium():
    spec = importlib.util.spec_from_file_location('ssd', ROOT/'tools/steady_state_rf_dnp.py')
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    cfg = MaterialConfig.load(material_path('demo_single_pake')); cfg.parameters.update(dnp_enabled=True, p_dnp_sat=0.5)
    m = cfg.make_model(preserve_state=False)
    st = tool.Steady(FrozenDynamics(PopulationSnapshot.capture(m)))
    Q, P, g, y, res = st.value_and_grad(np.zeros(len(m.mu)), np.r_[m.n[:, 0], m.n[:, 2]])
    assert res < 1e-15 and P == pytest.approx(0.5, abs=1e-12) and Q == pytest.approx(m.polarizations()['Q'], abs=1e-12)
