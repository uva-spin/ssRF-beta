"""Training-data generator: shard format, determinism, resume and export round trip."""
import json

import numpy as np
import pytest

from generate_training_data import default_workers, physical_cpu_count, plot_examples
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
    assert d['manipulation'][0] == 'ssrf'
    assert np.all(d['powers'] >= 0) and d['duration'][0] > 0
    assert 0.2 <= d['p0'][0] <= 0.25
    assert d['post_spectra'].shape == (1, 3, N) and np.allclose(d['post_times'][0], [10/3, 20/3, 10])
    assert d['post_Q'].shape == (1, 3)
    assert abs(d['state_Q'][0]-d['no_rf_Q'][0]) < 0.05 and d['final_Q'][0] >= d['no_rf_Q'][0]-1e-6
    plot_examples(tmp_path)
    assert (tmp_path/'example.png').stat().st_size > 0
    datagen.export_sample(tmp_path, 0, tmp_path/'p.json', tmp_path/'s.npz')
    s = PopulationSnapshot.load(tmp_path/'s.npz'); p = PulseProgram.load(tmp_path/'p.json')
    assert abs(polarizations(s.n)['Q']-d['state_Q'][0]) < 1e-6
    n, _ = replay(s, p, float(d['duration'][0]), dt=float(d['playback_dt'][0]))
    assert abs(polarizations(n)['Q']-d['final_Q'][0]) < 1e-5


def test_config_validation():
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


def test_afp_profile_stores_the_post_flip_spectrum(tmp_path):
    cfg = datagen.merged_config({'design': {'ssrf': False, 'afp_profile': True, 'preset': 'test'},
                                 'p0_range': [0.45, 0.45], 'history': {'n_burns': [0, 0], 'afp_probability': 0.0}})
    datagen.write_manifest(tmp_path, cfg, seed=1, n_samples=1, shard_size=1)
    r = datagen.run_block((str(tmp_path), 0, 1, 1, cfg))
    assert r['written'] == 1 and not r['failures']
    d = datagen.load_dataset(tmp_path)
    assert d['status'][0] == 'afp_profile' and d['manipulation'][0] == 'afp_profile' and d['duration'][0] == 0
    assert set(np.unique(d['powers'][0])).issubset({0.0, 1.0}) and d['powers'][0].sum() >= 1
    assert not np.allclose(d['spectrum'][0], d['final_spectrum'][0])
    n = d['populations'][0].astype(float)
    assert abs((n[:, 0]-n[:, 2]).sum()-d['final_P'][0]) < 1e-5
    assert abs((n[:, 0]-2*n[:, 1]+n[:, 2]).sum()-d['final_Q'][0]) < 1e-5
    assert json.loads(d['events_json'][0])[0]['kind'] == 'afp_profile'


def test_ssrf_and_afp_share_one_input_event(tmp_path):
    cfg = datagen.merged_config({'design': {'ssrf': True, 'afp_profile': True, 'preset': 'test'},
                                 'p0_range': [0.45, 0.45], 'history': {'n_burns': [0, 0], 'afp_probability': 0.0}})
    datagen.write_manifest(tmp_path, cfg, seed=1, n_samples=2, shard_size=2)
    r = datagen.run_block((str(tmp_path), 0, 2, 1, cfg))
    assert r['written'] == 2 and not r['failures']
    d = datagen.load_dataset(tmp_path)
    assert list(d['manipulation']) == ['ssrf', 'afp_profile']
    assert list(d['group']) == [0, 0] and np.allclose(d['spectrum'][0], d['spectrum'][1])
    assert not np.allclose(d['final_spectrum'][0], d['final_spectrum'][1])
    assert d['duration'][1] == 0 and d['powers'][1].sum() >= 1


def test_states_only_mode_skips_the_optimizer(tmp_path):
    cfg = datagen.merged_config({'design': {'skip': True}, 'history': {'n_burns': [0, 1], 'wait': [0.0, 1.0]}})
    datagen.write_manifest(tmp_path, cfg, seed=5, n_samples=3, shard_size=3)
    r = datagen.run_block((str(tmp_path), 0, 3, 5, cfg))
    assert r['written'] == 3 and not r['failures']
    d = datagen.load_dataset(tmp_path)
    assert set(d['status']) == {'state_only'} and set(d['manipulation']) == {'states_only'}
    assert np.all(d['powers'] == 0) and np.all(np.isnan(d['duration']))
    # labels are consistent with the stored populations
    for i in range(3):
        n = d['populations'][i].astype(float)
        assert abs((n[:, 0]-n[:, 2]).sum()-d['state_P'][i]) < 1e-5
        assert abs((n[:, 0]-2*n[:, 1]+n[:, 2]).sum()-d['state_Q'][i]) < 1e-5


def test_chain_mode_rounds_follow_applied_programs(tmp_path):
    """Chain mode: each round's input is the state after the previous (undershot) program."""
    cfg = datagen.merged_config({'design': {'preset': 'test'}, 'p0_range': [0.25, 0.3],
                                 'history': {'n_burns': [0, 0], 'afp_probability': 0.0},
                                 'chain': {'n_chains': 1, 'rounds': 3, 'wait': [1.0, 2.0],
                                           'wait_zero_probability': 0.0, 'exact_probability': 0.0}})
    datagen.write_manifest(tmp_path, cfg, seed=5, n_samples=3, shard_size=1)
    r = datagen.run_block((str(tmp_path), 0, 1, 5, cfg))
    assert r['written'] == 3 and not r['failures']
    d = datagen.load_dataset(tmp_path)
    assert list(d['chain_round']) == [0, 1, 2] and np.all(d['group'] == 0)
    assert np.all((d['time_since_burn'] >= 1.0) & (d['time_since_burn'] <= 2.0))
    ev = json.loads(d['events_json'][2])
    applied = [e for e in ev if e['kind'] == 'applied_design']
    assert len(applied) == 2 and all(e['total_power_U'] <= e['optimal_total_power_U']+1e-9 for e in applied)
    assert all(e['duration'] <= e['optimal_duration']+1e-9 for e in applied)
    assert not np.allclose(d['spectrum'][0], d['spectrum'][1])     # the applied pulse changed the state


def test_default_workers_use_physical_cores():
    n = physical_cpu_count()
    assert 1 <= default_workers() <= n
    assert default_workers() == max(1, n - 1)


def test_fast_preset_is_a_short_search():
    p = datagen.OPTIMIZER_PRESETS['fast']
    assert p['max_iterations'] == 8 and p['starts'] == 1 and p['search_dt'] == 0.02


def test_run_for_batches_whole_steps():
    mat = MaterialConfig.load(material_path('d_butanol_matched'))
    a = mat.make_model(preserve_state=False)
    b = MaterialConfig.from_dict(mat.to_dict()).make_model(preserve_state=False)
    datagen._run_for(a, 1.0, False, False)
    b.params.rf_enabled = False
    b.params.dnp_enabled = False
    end = b.t + 1.0
    dt = float(b.params.dt)
    while b.t < end - 1e-12:
        b.params.dt = min(dt, end - b.t)
        b.step(1)
    b.params.dt = dt
    np.testing.assert_allclose(a.n, b.n)
    assert a.t == pytest.approx(b.t)


def test_undershoot_respects_caps():
    rng = np.random.default_rng(0)
    U = np.r_[np.zeros(10), np.full(5, 0.4)]
    for _ in range(50):
        V, T, info = datagen.undershoot(U, 30.0, datagen.CHAIN_DEFAULTS, rng)
        assert V.sum() <= U.sum()+1e-12 and T <= 30.0 and V.max() <= U.max()+1e-12
        assert np.all(V[:10] == 0)
