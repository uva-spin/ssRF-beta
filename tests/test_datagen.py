"""Training-data generator: shard format, determinism, resume and export round trip."""
import json

import numpy as np

from ssrf_realtime import datagen
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot
from ssrf_realtime.presets import material_path
from ssrf_realtime.pulse_program import PulseProgram
from ssrf_realtime.tensor_optimizer import polarizations, replay

CFG = datagen.merged_config({'design': {'preset': 'test'}, 'p0_range': [0.2, 0.25],
                             'history': {'n_burns': [1, 1], 'wait': [1.0, 5.0], 'afp_probability': 1.0},
                             'post': {'time': 10.0, 'steps': 3, 'dnp': False}})


def test_history_is_deterministic_and_changes_the_state():
    mat = MaterialConfig.load(material_path('d_butanol_matched'))
    a, ea = datagen.generate_state(MaterialConfig.from_dict(mat.to_dict()), CFG['history'], np.random.default_rng([1, 7]))
    b, eb = datagen.generate_state(MaterialConfig.from_dict(mat.to_dict()), CFG['history'], np.random.default_rng([1, 7]))
    assert ea == eb and np.array_equal(a.n, b.n)
    assert {e['kind'] for e in ea} >= {'burn', 'afp'}
    fresh = mat.make_model(preserve_state=False)
    assert not np.allclose(a.n, fresh.n)
    assert np.allclose(a.n.sum(axis=1), a.mu)


def test_shard_manifest_resume_and_export_round_trip(tmp_path):
    datagen.write_manifest(tmp_path, CFG, seed=3, n_samples=1, shard_size=1)
    r = datagen.run_block((str(tmp_path), 0, 1, 3, CFG))
    assert r['written'] == 1 and not r['failures']
    assert datagen.run_block((str(tmp_path), 0, 1, 3, CFG))['skipped']          # resume skips it
    m = json.loads((tmp_path/'manifest.json').read_text())
    d = datagen.load_dataset(tmp_path)
    N = m['grid']['n_bins']
    assert d['spectrum'].shape == (1, N) and d['powers'].shape == (1, N) and d['populations'].shape == (1, N, 3)
    assert np.all(d['powers'] >= 0) and d['duration'][0] > 0
    assert 0.2 <= d['p0'][0] <= 0.25
    assert d['post_spectra'].shape == (1, 3, N) and np.allclose(d['post_times'][0], [10/3, 20/3, 10])
    assert d['post_Q'].shape == (1, 3)
    assert abs(d['state_Q'][0]-d['no_rf_Q'][0]) < 0.05 and d['final_Q'][0] >= d['no_rf_Q'][0]-1e-6
    datagen.export_sample(tmp_path, 0, tmp_path/'p.json', tmp_path/'s.npz')
    s = PopulationSnapshot.load(tmp_path/'s.npz'); p = PulseProgram.load(tmp_path/'p.json')
    assert abs(polarizations(s.n)['Q']-d['state_Q'][0]) < 1e-6
    n, _ = replay(s, p, float(d['duration'][0]), dt=float(d['playback_dt'][0]))
    assert abs(polarizations(n)['Q']-d['final_Q'][0]) < 1e-5


def test_config_validation():
    import pytest
    with pytest.raises(ValueError):
        datagen.merged_config({'post': {'time': 5.0, 'steps': 0}})
    with pytest.raises(ValueError):
        datagen.merged_config({'p0_range': [0.6, 0.2]})


def test_replay_from_late_snapshot_with_tiny_step():
    """Regression: replay at large simulation time must not stall on float resolution."""
    mat = MaterialConfig.load(material_path('d_butanol_matched'))
    m = mat.make_model(preserve_state=False); m.t = 987.654321
    s = PopulationSnapshot.capture(m)
    R = m.Rplus; u = np.zeros(len(R)); u[np.abs(R+0.93) < 0.05] = 5.0
    from ssrf_realtime.tensor_optimizer import synchronized_program
    n, _ = replay(s, synchronized_program(R, u, 0.0237137), 0.0237137, dt=1.46484375e-05)
    assert np.isfinite(n).all() and polarizations(n)['Q'] > polarizations(m.n)['Q']


def test_states_only_mode_skips_the_optimizer(tmp_path):
    cfg = datagen.merged_config({'design': {'skip': True}, 'history': {'n_burns': [0, 1], 'wait': [0.0, 1.0]}})
    datagen.write_manifest(tmp_path, cfg, seed=5, n_samples=3, shard_size=3)
    r = datagen.run_block((str(tmp_path), 0, 3, 5, cfg))
    assert r['written'] == 3 and not r['failures']
    d = datagen.load_dataset(tmp_path)
    assert set(d['status']) == {'state_only'} and np.all(d['powers'] == 0) and np.all(np.isnan(d['duration']))
    # labels are consistent with the stored populations
    for i in range(3):
        n = d['populations'][i].astype(float)
        assert abs((n[:, 0]-n[:, 2]).sum()-d['state_P'][i]) < 1e-5
        assert abs((n[:, 0]-2*n[:, 1]+n[:, 2]).sum()-d['state_Q'][i]) < 1e-5
