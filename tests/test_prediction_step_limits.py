"""Regressions for the high-gain full-search failure and duration-specific caps."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.material_config import PopulationSnapshot, MaterialConfig
from fixture_materials import default_material
from ssrf_realtime.tensor_optimizer import (
    FrozenDynamics, OptimizerSettings, candidate_arrays, prediction_step_plan,
    optimize_tensor, replay,
)


def test_shipped_default_full_frequency_range_fits_the_prediction_limit():
    config=default_material()
    settings=OptimizerSettings.from_dict(config.optimizer)
    dyn=FrozenDynamics(PopulationSnapshot.capture(config.make_model(preserve_state=False)))
    _,upper,_,_=candidate_arrays(dyn,settings)
    plan=prediction_step_plan(dyn,upper,settings,settings.max_duration)
    assert settings.regions==[] and config.parameters['rf_response_mode']=='spectral'
    assert max(plan['search_steps'],plan['endpoint_steps'])<=settings.max_steps


@pytest.mark.parametrize('bins,required', [(500,'24,981'),(701,'24,997')])
def test_previous_gain100_full_search_reports_required_steps_and_recovery_options(bins,required):
    config=MaterialConfig.load(Path(__file__).resolve().parents[1]/'tests/fixtures/materials/pedestal_easy.json')
    config.parameters['n_bins']=bins
    config.parameters['rf_pedestal_gain']=100.
    snapshot=PopulationSnapshot.capture(config.make_model(preserve_state=False))
    settings=OptimizerSettings.from_dict(config.optimizer)
    with pytest.raises(ValueError) as exc:
        optimize_tensor(snapshot,settings)
    message=str(exc.value)
    assert required in message and '20,000' in message
    assert 'Maximum prediction steps' in message and 'Pedestal RF gain' in message


def test_fine_endpoint_step_limit_is_checked_before_starting_the_search():
    model=IdealBinModel(IdealBinParams(n_bins=31,dt=1e-6,diffusion_enabled=False))
    snapshot=PopulationSnapshot.capture(model)
    settings=OptimizerSettings(max_power=1,min_duration=.01,max_duration=.2)
    with pytest.raises(ValueError,match='200,001|200,000') as exc:
        optimize_tensor(snapshot,settings)
    assert 'endpoint' in str(exc.value) and 'material dt' in str(exc.value)
    np.testing.assert_array_equal(model.n,snapshot.n)


def test_long_budgeted_horizons_do_not_inherit_short_horizon_power_ceilings():
    model=IdealBinModel(IdealBinParams(n_bins=31,rf_response_mode='independent',
        rf_pedestal_gain=100.,diffusion_enabled=False))
    snapshot=PopulationSnapshot.capture(model)
    settings=OptimizerSettings(max_power=20,min_duration=.01,max_duration=2.,
        duration_samples=2,duration_refinements=0,starts=1,max_iterations=3,
        max_steps=2000,max_wall_seconds=30,max_total_exposure=.1)
    dyn=FrozenDynamics(snapshot);_,upper,_,_=candidate_arrays(dyn,settings)
    # Reproduce the old mixed-duration preflight: it wrongly rejected this cap.
    old_upper=np.minimum(upper,settings.max_total_exposure/settings.min_duration)
    old_dt=min(settings.search_dt,.18/dyn.outgoing_bound(old_upper))
    assert np.ceil(settings.max_duration/old_dt)>settings.max_steps
    result=optimize_tensor(snapshot,settings)
    assert len(result.scan)==2
    assert result.scan[1]['search_dt']>result.scan[0]['search_dt']
    assert all(max(row['search_steps'],row['endpoint_steps'])<=settings.max_steps for row in result.scan)
    assert result.report['status']=='completed'
    assert result.report['rf_exposure_budget']['used']<=settings.max_total_exposure
    actual,_=replay(snapshot,result.program,result.duration,dt=result.report['recommended_playback_dt'])
    np.testing.assert_array_equal(actual,result.final_n)
    np.testing.assert_array_equal(snapshot.n,model.n)


def test_nonbinding_exposure_cap_keeps_the_same_step_plan():
    dyn=FrozenDynamics(PopulationSnapshot.capture(IdealBinModel(IdealBinParams(n_bins=31))))
    settings=OptimizerSettings();_,upper,_,_=candidate_arrays(dyn,settings)
    for duration in (.01,.2,2.):
        assert prediction_step_plan(dyn,upper,settings,duration)==prediction_step_plan(
            dyn,upper,replace(settings,max_total_exposure=1e6),duration)
