"""Total instantaneous RF power must not grow with the number of carriers."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest

from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot
from ssrf_realtime.pulse_program import PulseProgram
from ssrf_realtime.tensor_optimizer import (OptimizerSettings, effective_exposure_limit,
                                           optimize_tensor, replay)


@pytest.mark.parametrize('value', [-1, True, '1', float('nan'), float('inf')])
def test_invalid_total_power(value):
    with pytest.raises(ValueError, match='max_total_power'):
        OptimizerSettings(max_total_power=value).validate()


def test_old_settings_remain_unlimited_and_caps_combine_at_each_duration():
    old = asdict(OptimizerSettings())
    del old['max_total_power']
    assert OptimizerSettings.from_dict(old).max_total_power is None
    settings = OptimizerSettings(max_total_power=2., max_total_exposure=3.)
    assert effective_exposure_limit(settings, .1) == pytest.approx(.2)
    assert effective_exposure_limit(settings, 4.) == 3.
    settings.max_total_exposure = None
    assert effective_exposure_limit(settings, 4.) == 8.


@pytest.mark.parametrize('power,exposure', [(0., None), (.3, None), (.3, .01)])
def test_power_and_exposure_limits_survive_duration_scan_export_and_replay(tmp_path, power, exposure):
    m = IdealBinModel(IdealBinParams(n_bins=41, diffusion_enabled=False))
    snapshot = PopulationSnapshot.capture(m)
    settings = OptimizerSettings(max_power=5., max_total_power=power, max_total_exposure=exposure,
        min_duration=.03, max_duration=.15, duration_samples=3, duration_refinements=1,
        starts=1, max_iterations=25, max_wall_seconds=30)
    result = optimize_tensor(snapshot, settings)
    assert result.powers.sum() <= power*(1+1e-12)
    assert result.report['rf_total_power']['satisfied']
    for row in result.scan:
        assert row['integrated_base_exposure'] <= power*row['duration']*(1+1e-12)
        if exposure is not None:
            assert row['integrated_base_exposure'] <= exposure*(1+1e-12)
    result.export(tmp_path)
    loaded = MaterialConfig.load(tmp_path/'material.json')
    assert loaded.optimizer['max_total_power'] == power
    program = PulseProgram.load(tmp_path/'rf_program.json')
    assert program.compile().envelope.sum() <= power*(1+1e-12)
    n, _ = replay(snapshot, program, result.duration, dt=result.report['recommended_playback_dt'])
    np.testing.assert_allclose(n, result.final_n, atol=2e-16)


def test_nonbinding_power_cap_preserves_the_original_search():
    snapshot = PopulationSnapshot.capture(IdealBinModel(IdealBinParams(n_bins=31)))
    settings = dict(min_duration=.03, max_duration=.06, duration_samples=2,
                    duration_refinements=0, max_iterations=10, starts=1)
    original = optimize_tensor(snapshot, OptimizerSettings(**settings))
    capped = optimize_tensor(snapshot, OptimizerSettings(max_total_power=1e6, **settings))
    np.testing.assert_array_equal(original.powers, capped.powers)
    np.testing.assert_array_equal(original.final_n, capped.final_n)


def test_experimental_reference_has_one_total_command_and_separate_energy_budget():
    root = Path(__file__).resolve().parents[1]
    config = MaterialConfig.load(root/'tests/fixtures/materials/experimental_event_1406922415.json')
    program = PulseProgram.load(root/'tests/fixtures/programs/experimental_event_1406922415.json')
    settings = OptimizerSettings.from_dict(config.optimizer)
    assert settings.max_total_power == 1.
    assert settings.max_total_exposure == 34.
    assert program.compile().envelope.sum() == pytest.approx(1., abs=1e-10)
    assert program.compile().exposure.sum() == pytest.approx(34., abs=1e-9)


def test_experimental_reference_burns_about_80_percent_in_three_time_units():
    root = Path(__file__).resolve().parents[1]
    config = MaterialConfig.load(root/'tests/fixtures/materials/experimental_event_1406922415.json')
    model = config.make_model(preserve_state=False)
    model.set_program(PulseProgram.load(root/'tests/fixtures/programs/experimental_event_1406922415.json'))
    peak = int(np.argmin(abs(model.Rplus+.96)))
    before = model.spectrum()[3][peak]
    model.start_program()
    model.step(100)
    assert model.t == pytest.approx(3.)
    remaining = model.spectrum()[3][peak]/before
    assert .17 < remaining < .25
    assert model.delivered_exposure.sum() == pytest.approx(3., abs=1e-9)
    assert not model.params.dnp_enabled
