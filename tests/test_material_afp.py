"""AFP persistence, legacy-file migration, and post-AFP tensor design."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.material_config import (
    MaterialConfig, PopulationSnapshot, _state_digest, validate_params,
)
from ssrf_realtime.pulse_program import PulseProgram
from ssrf_realtime.tensor_optimizer import OptimizerSettings, optimize_tensor, replay


def model(**kwargs):
    return IdealBinModel(IdealBinParams(n_bins=31, **kwargs))


@pytest.mark.parametrize('subset', [None, [], [3, 6, 15]])
def test_material_roundtrip_preserves_afp_settings_and_live_state(tmp_path, subset):
    m = model(afp_efficiency=0.65, afp_subset_indices=subset)
    m.afp_sweep()
    m.step(2, rf_on=False)
    config = MaterialConfig.capture(m)
    path = tmp_path / 'material.json'
    config.save(path)
    loaded = MaterialConfig.load(path)
    assert loaded.parameters == asdict(m.params)
    assert loaded.to_dict()['parameters']['afp'] == {
        'afp_efficiency': 0.65, 'afp_subset_indices': subset,
    }
    restored = loaded.make_model(m, preserve_state=True)
    np.testing.assert_array_equal(restored.n, m.n)
    np.testing.assert_array_equal(restored.n_ref, m.n_ref)
    assert restored.t == m.t
    assert not restored.params.rf_enabled
    assert restored.program_state == 'stopped'


@pytest.mark.parametrize('filename', ['default_demo.json', 'negative_demo.json'])
def test_bundled_legacy_material_loads_with_afp_defaults(filename):
    path = Path(__file__).resolve().parents[1] / 'tests/fixtures/materials' / filename
    loaded = MaterialConfig.load(path)
    restored = loaded.make_model(preserve_state=False)
    assert restored.params.afp_efficiency == 1.0
    assert restored.params.afp_subset_indices is None
    assert MaterialConfig.from_dict(loaded.to_dict()).parameters == loaded.parameters


@pytest.mark.parametrize('key,value', [
    ('afp_efficiency', -0.1), ('afp_efficiency', 1.1),
    ('afp_efficiency', float('nan')), ('afp_efficiency', float('inf')),
    ('afp_efficiency', True), ('afp_efficiency', '1'),
    ('afp_subset_indices', 4), ('afp_subset_indices', '4'),
    ('afp_subset_indices', [True]), ('afp_subset_indices', [1.5]),
    ('afp_subset_indices', [-1]), ('afp_subset_indices', [31]),
    ('afp_subset_indices', [3, 3]), ('afp_subset_indices', [3, 27]),
])
def test_invalid_afp_settings_are_rejected(key, value):
    parameters = asdict(IdealBinParams(n_bins=31))
    parameters[key] = value
    with pytest.raises(ValueError, match=key):
        validate_params(parameters)


@pytest.mark.parametrize('change', ['partial_afp', 'unknown_afp', 'missing_grid', 'unknown_group'])
def test_legacy_compatibility_does_not_weaken_material_schema(change):
    data = MaterialConfig.capture(model()).to_dict()
    groups = data['parameters']
    if change == 'partial_afp':
        del groups['afp']['afp_efficiency']
    elif change == 'unknown_afp':
        groups['afp']['typo'] = 1
    elif change == 'missing_grid':
        del groups['grid']
    else:
        groups['typo'] = {}
    with pytest.raises(ValueError):
        MaterialConfig.from_dict(data)


@pytest.mark.parametrize('subset', [None, [], [4, 7]])
def test_snapshot_roundtrip_after_afp(tmp_path, subset):
    m = model(afp_efficiency=0.8, afp_subset_indices=subset)
    m.afp_sweep()
    m.step(2, rf_on=False)
    snapshot = PopulationSnapshot.capture(m)
    path = tmp_path / 'state.npz'
    snapshot.save(path)
    loaded = PopulationSnapshot.load(path)
    restored = loaded.make_model()
    np.testing.assert_array_equal(restored.n, m.n)
    np.testing.assert_array_equal(restored.n_ref, m.n_ref)
    assert restored.t == m.t
    assert restored.params.afp_efficiency == 0.8
    assert restored.params.afp_subset_indices == subset
    assert not restored.params.rf_enabled
    assert loaded.fingerprint == snapshot.fingerprint
    assert loaded.matches(restored)
    restored.afp_sweep([2])
    assert not loaded.matches(restored)


def legacy_snapshot():
    m = model()
    m.start_program()
    m.step(3)
    snapshot = PopulationSnapshot.capture(m)
    del snapshot.parameters['afp_efficiency']
    del snapshot.parameters['afp_subset_indices']
    snapshot.fingerprint = _state_digest(snapshot.parameters, snapshot.n, snapshot.mu)
    return snapshot


def test_legacy_snapshot_migrates_without_resetting_populations(tmp_path):
    original = legacy_snapshot()
    path = tmp_path / 'legacy.npz'
    original.save(path)
    loaded = PopulationSnapshot.load(path)
    restored = loaded.make_model()
    np.testing.assert_array_equal(restored.n, original.n)
    np.testing.assert_array_equal(restored.n_ref, original.n_ref)
    assert restored.t == original.simulation_time
    assert restored.params.afp_efficiency == 1.0
    assert restored.params.afp_subset_indices is None
    assert loaded.matches(restored)
    assert not restored.params.rf_enabled
    loaded.save(tmp_path / 'migrated.npz')
    assert PopulationSnapshot.load(tmp_path / 'migrated.npz').fingerprint == loaded.fingerprint


@pytest.mark.parametrize('legacy', [False, True])
def test_snapshot_migration_rejects_corrupted_fingerprint(tmp_path, legacy):
    snapshot = legacy_snapshot() if legacy else PopulationSnapshot.capture(model())
    snapshot.parameters['diffusion_scale'] += 1
    path = tmp_path / 'corrupted.npz'
    snapshot.save(path)
    with pytest.raises(ValueError, match='fingerprint'):
        PopulationSnapshot.load(path)


def test_design_export_and_replay_from_post_afp_snapshot(tmp_path):
    m = model(afp_efficiency=0.6)
    m.afp_sweep([10])
    before = m.n.copy()
    snapshot = PopulationSnapshot.capture(m)
    settings = OptimizerSettings(
        max_power=4, min_duration=0.02, max_duration=0.05,
        duration_samples=2, duration_refinements=0,
        max_iterations=6, starts=1, max_wall_seconds=30,
    )
    result = optimize_tensor(snapshot, settings)
    assert result.report['status'] == 'completed'
    np.testing.assert_array_equal(m.n, before)
    directory = tmp_path / 'design'
    result.export(directory)
    loaded = PopulationSnapshot.load(directory / 'starting_state.npz')
    material = MaterialConfig.load(directory / 'material.json')
    assert material.parameters == snapshot.parameters
    program = PulseProgram.load(directory / 'rf_program.json')
    endpoint, _ = replay(loaded, program, result.duration,
                         dt=result.report['recommended_playback_dt'])
    np.testing.assert_allclose(endpoint, result.final_n, rtol=2e-13, atol=1e-16)
