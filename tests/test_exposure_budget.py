"""Total RF budget constraints, reallocation quality, persistence and replay."""
from dataclasses import asdict

import numpy as np
import pytest

from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot
from ssrf_realtime.pulse_program import PulseProgram
from ssrf_realtime.tensor_optimizer import (
    OptimizerSettings, optimize_tensor, replay, synchronized_program,
)


def settings(**kwargs):
    values=dict(max_power=20, min_duration=.2, max_duration=.2,
                duration_samples=2, duration_refinements=0, starts=1,
                max_iterations=60, max_wall_seconds=30)
    values.update(kwargs)
    return OptimizerSettings(**values)


def snapshot(**kwargs):
    return PopulationSnapshot.capture(IdealBinModel(IdealBinParams(n_bins=101, **kwargs)))


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '5'])
def test_invalid_exposure_budget(value):
    with pytest.raises(ValueError, match='max_total_exposure'):
        settings(max_total_exposure=value).validate()


def test_old_settings_and_materials_default_to_unlimited(tmp_path):
    old=asdict(settings())
    del old['max_total_exposure']
    assert OptimizerSettings.from_dict(old).max_total_exposure is None
    data=MaterialConfig.capture(snapshot().make_model()).to_dict()
    del data['optimizer']['max_total_exposure']
    config=MaterialConfig.from_dict(data)
    assert OptimizerSettings.from_dict(config.optimizer).max_total_exposure is None
    config.optimizer=asdict(settings(max_total_exposure=3.25))
    config.save(tmp_path/'material.json')
    loaded=MaterialConfig.load(tmp_path/'material.json')
    assert OptimizerSettings.from_dict(loaded.optimizer).max_total_exposure == 3.25
    old['typo']=2
    with pytest.raises(ValueError):
        OptimizerSettings.from_dict(old)


@pytest.fixture(scope='module')
def unrestricted():
    s=snapshot(diffusion_enabled=False)
    return s, optimize_tensor(s,settings())


def test_nonbinding_budget_preserves_unlimited_result(unrestricted):
    s, original=unrestricted
    limited=optimize_tensor(s,settings(max_total_exposure=1e6))
    np.testing.assert_array_equal(limited.powers, original.powers)
    np.testing.assert_array_equal(limited.final_n, original.final_n)
    assert limited.duration == original.duration
    assert limited.scan[0]['solver'] == 'L-BFGS-B'


@pytest.mark.parametrize('fraction', [.5, .25, .1, 1e-6])
def test_budget_reallocates_RF_and_is_respected_in_export_and_playback(unrestricted, fraction, tmp_path):
    s, original=unrestricted
    limit=original.report['integrated_base_exposure']*fraction
    result=optimize_tensor(s,settings(max_total_exposure=limit))
    assert result.report['rf_exposure_budget']['satisfied']
    assert result.report['integrated_base_exposure'] <= limit*(1+1e-12)
    assert all(row['integrated_base_exposure'] <= limit*(1+1e-12) for row in result.scan)
    assert np.all(result.powers <= result.settings.max_power)
    assert not result.powers[~result.candidates].any()
    result.export(tmp_path)
    loaded=PopulationSnapshot.load(tmp_path/'starting_state.npz')
    for program in (PulseProgram.load(tmp_path/'rf_program.json'),
                    PulseProgram.load_csv(tmp_path/'rf_program.csv')):
        assert program.compile().exposure.sum() <= limit*(1+1e-12)
        n,_=replay(loaded,program,result.duration,dt=result.report['recommended_playback_dt'])
        np.testing.assert_allclose(n,result.final_n,rtol=2e-13,atol=1e-16)
    material=MaterialConfig.load(tmp_path/'material.json')
    assert material.optimizer['max_total_exposure'] == limit
    # Compare to uniform attenuation of the old pulse at the same expenditure.
    scaled=synchronized_program(s.grid,original.powers*fraction,original.duration)
    scaled_n,_=replay(s,scaled,original.duration,dt=result.report['recommended_playback_dt'])
    scaled_Q=float(np.sum(scaled_n[:,0]-2*scaled_n[:,1]+scaled_n[:,2]))
    assert result.report['final']['Q'] >= scaled_Q-2e-6
    if fraction == .5:
        assert result.report['delta_Q'] > .9*original.report['delta_Q']
        assert np.max(abs(result.powers-original.powers*fraction)) > .1
    # Actual scheduled delivery uses the same single physical-bin command cost.
    model=s.make_model()
    if result.duration:
        model.set_program(result.program); model.start_program()
        while model.program_state != 'finished':
            model.step()
        assert model.delivered_exposure.sum() <= limit*(1+1e-10)


@pytest.mark.parametrize('natural', [False, True])
def test_zero_budget_forbids_rf_but_allows_natural_dynamics(natural):
    s=snapshot(diffusion_enabled=False,dnp_enabled=natural)
    result=optimize_tensor(s,settings(max_total_exposure=0))
    assert not result.powers.any()
    assert result.program.compile().exposure.sum() == 0
    assert result.report['rf_exposure_budget']['used'] == 0
    assert result.report['rf_exposure_budget']['fraction_used'] is None
    # Scheduler replay and the independent no-RF solve use different step grids.
    assert abs(result.report['RF_advantage_over_no_RF']) <= result.settings.convergence_tolerance
    if natural:
        assert result.report['status'] == 'no_RF_benefit'
        assert result.report['delta_Q'] > 0
    else:
        assert result.report['status'] == 'no_improvement'
        np.testing.assert_array_equal(result.final_n,s.n)


@pytest.mark.parametrize('p0', [.45,-.45])
def test_budget_applies_at_every_duration_with_afp_and_full_dynamics(p0):
    model=IdealBinModel(IdealBinParams(n_bins=41,p0=p0,afp_efficiency=.6,dnp_enabled=True,t1_rate=.02))
    model.afp_sweep([12])
    before=model.n.copy(); time=model.t
    s=PopulationSnapshot.capture(model)
    result=optimize_tensor(s,settings(min_duration=.02,max_duration=.12,
        duration_samples=3,duration_refinements=1,max_total_exposure=.7,
        per_bin_power_limits={'12':0.,'13':.5}))
    assert result.report['integrated_base_exposure'] <= .7*(1+1e-12)
    assert len(result.scan) >= 3
    assert all(row['integrated_base_exposure'] <= .7*(1+1e-12) for row in result.scan)
    assert result.powers[12] == 0 and result.powers[13] <= .5
    np.testing.assert_array_equal(model.n,before)
    assert model.t == time


def test_iteration_limit_still_returns_only_feasible_controls(unrestricted):
    s, _=unrestricted
    result=optimize_tensor(s,settings(max_total_exposure=.2,max_iterations=1))
    assert all(row['integrated_base_exposure'] <= .2*(1+1e-12) for row in result.scan)
    assert result.report['rf_exposure_budget']['used'] <= .2*(1+1e-12)
