"""Training-data generation: (manipulated/evolved spectrum) -> (Q-optimizing RF program).

Each sample:
  1. Draws a material from the configured list and a Boltzmann start at a random P.
  2. Applies a random history with the live simulator: RF burns (random window,
     power, duration), optional AFP windows, and free evolution (spin diffusion,
     DNP on or off) between them.
  3. Records the resulting spectrum: the total signal I+ + I- (what the
     instrument measures), and optionally I+ and I- separately and the full
     level populations.
  4. Runs the tensor optimizer from that exact state and records the optimal
     synchronized program: per-bin RF power U_j and common duration T. Also
     records the Q and P it reaches, the no-RF control and the optimizer status.

Everything is deterministic given (global seed, sample id). Samples are written
in shards (one .npz per block of ids), so a run can be resumed and parallelized.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .afp import mirror_safe_indices
from .material_config import MaterialConfig, PopulationSnapshot
from .presets import material_path
from .pulse_program import BinPulse, PulseProgram, RFProfile
from .tensor_optimizer import OptimizerSettings, optimize_tensor, synchronized_program

SCHEMA = 'ssrf.training_data.v1'

# --------------------------------------------------------------------------- configuration
DEFAULT_CONFIG: Dict = {
    'materials': ['d_butanol_matched'],
    'p0_range': [0.30, 0.60],                 # Boltzmann start, P drawn uniformly
    'history': {
        'n_burns': [0, 3],                    # inclusive integer range
        'burn_center_abs_R': [0.0, 2.2],      # |R| of the window centre (sign random)
        'burn_width_R': [0.02, 0.6],          # log-uniform
        'burn_total_power_U': [0.01, 5.0],    # log-uniform, spread evenly over the window bins
        'burn_duration': [0.01, 20.0],        # log-uniform (simulation time units)
        'afp_probability': 0.2,               # chance of one AFP window in the history
        'afp_width_R': [0.05, 0.6],
        'afp_efficiency': [0.7, 1.0],
        'wait': [0.0, 600.0],                 # free evolution after each event, log-uniform above 0.1
        'wait_zero_probability': 0.2,         # chance an event is followed by no wait
        'dnp_on_probability': 0.3,            # DNP state during each wait
        'dnp_during_burns': False,            # DNP state during history RF burns
    },
    'design': {
        'preset': 'fast',                     # fast | standard | full (see OPTIMIZER_PRESETS)
        'dnp_during_pulse': False,            # DNP state assumed by the optimizer
        'max_total_power': None,              # None = uncapped; 1 = 0.1 W
        'max_total_exposure': None,           # None = uncapped; 34 = 3.4 J
        'overrides': {},                      # any OptimizerSettings field
    },
    'sweep': None,                            # e.g. {'p0_values': [...], 'histories_per_p0': 8, 'times': [...], 'dnp': True}
    'post': {
        'time': 0.0,                          # free evolution after the optimized pulse (0 = none)
        'steps': 0,                           # number of evenly spaced snapshots recorded over 'time'
        'dnp': False,                         # DNP on during the post-pulse evolution
    },
    'store': {'branches': True, 'populations': True},
}

# Overrides applied on top of the material's own optimizer settings.
OPTIMIZER_PRESETS = {
    # Quick targets for large datasets: coarser duration scan, one start, no smoothing pass.
    'fast': dict(duration_samples=5, duration_refinements=1, max_iterations=40, starts=1,
                 profile_smoothing_fwhm_R=0.0, max_wall_seconds=180.0, max_validation_halvings=12),
    # The material's settings with a bounded wall time.
    'standard': dict(max_wall_seconds=600.0, max_validation_halvings=12),
    # As in the GUI, with a generous wall time.
    'full': dict(max_wall_seconds=1800.0, max_validation_halvings=12),
    # For tests only.
    'test': dict(duration_samples=2, duration_refinements=0, max_iterations=3, starts=1,
                 profile_smoothing_fwhm_R=0.0, max_wall_seconds=60.0, max_validation_halvings=12),
}


def merged_config(user: Optional[Dict] = None) -> Dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, value in (user or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    lo, hi = cfg['p0_range']
    if not (-0.999 < lo <= hi < 0.999):
        raise ValueError('p0_range must satisfy -0.999 < min <= max < 0.999 (fractions)')
    if cfg['post']['steps'] < 0 or cfg['post']['time'] < 0 or (cfg['post']['steps'] > 0) != (cfg['post']['time'] > 0):
        raise ValueError('post: give both time > 0 and steps > 0, or neither')
    sw = cfg.get('sweep')
    if sw:
        if not sw.get('p0_values') or int(sw.get('histories_per_p0', 0)) < 1 or not sw.get('times'):
            raise ValueError('sweep needs p0_values, histories_per_p0 >= 1 and times')
        if min(sw['times']) < 0 or any(not -0.999 < p < 0.999 for p in sw['p0_values']):
            raise ValueError('sweep times must be >= 0 and p0_values fractions in (-1, 1)')
    if cfg['design']['preset'] not in OPTIMIZER_PRESETS:
        raise ValueError(f"Unknown preset {cfg['design']['preset']!r}; choose {sorted(OPTIMIZER_PRESETS)}")
    return cfg


# --------------------------------------------------------------------------- random helpers
def _loguniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _window_bins(R, center, width):
    return np.flatnonzero(np.abs(R-center) <= width/2)


# --------------------------------------------------------------------------- state generation
def _run_for(model, duration, rf_on, dnp_on):
    """Advance the live model by `duration` with the given switches (exact pulse edges)."""
    model.params.rf_enabled = bool(rf_on)
    model.params.dnp_enabled = bool(dnp_on)
    end = model.t+float(duration); dt = float(model.params.dt)
    while model.t < end-1e-12:
        model.params.dt = min(dt, end-model.t)
        model.step(1)
    model.params.dt = dt


def generate_state(cfg_material: MaterialConfig, history: Dict, rng) -> (object, List[Dict]):
    """Return (model at the manipulated/evolved state, list of applied events)."""
    model = cfg_material.make_model(preserve_state=False)
    R = model.Rplus; N = len(R); events = []
    h = history
    n_burns = int(rng.integers(h['n_burns'][0], h['n_burns'][1]+1))
    afp_slot = int(rng.integers(0, n_burns+1)) if rng.random() < h['afp_probability'] else -1
    for k in range(n_burns+1):
        if k == afp_slot:
            width = _loguniform(rng, *h['afp_width_R'])
            side = rng.choice([-1.0, 1.0])
            center = side*rng.uniform(width/2, h['burn_center_abs_R'][1])
            bins = mirror_safe_indices(_window_bins(R, center, width), N)
            eff = float(rng.uniform(*h['afp_efficiency']))
            if bins:
                model.params.afp_efficiency = eff
                model.afp_sweep(bins)
                events.append(dict(kind='afp', center_R=float(center), width_R=float(width), efficiency=eff, bins=len(bins)))
        if k == n_burns:
            break
        center = rng.choice([-1.0, 1.0])*rng.uniform(*h['burn_center_abs_R'])
        width = _loguniform(rng, *h['burn_width_R'])
        bins = _window_bins(R, center, width)
        if len(bins) == 0:
            bins = np.array([int(np.argmin(abs(R-center)))])
        power = _loguniform(rng, *h['burn_total_power_U'])
        duration = _loguniform(rng, *h['burn_duration'])
        program = PulseProgram(N, float(R[0]), float(R[-1]), profiles=[RFProfile(
            'history burn', pulses=[BinPulse(int(j), power/len(bins), 0.0, duration) for j in bins])])
        model.set_program(program); model.start_program(True)
        _run_for(model, duration, rf_on=True, dnp_on=bool(h.get('dnp_during_burns', False)))
        model.stop_program()
        events.append(dict(kind='burn', center_R=float(center), width_R=float(width), total_power_U=power,
                           duration=duration, bins=int(len(bins))))
        if rng.random() >= h['wait_zero_probability']:
            lo, hi = float(h['wait'][0]), float(h['wait'][1])
            if hi <= 0:
                wait = 0.0
            elif lo > 0:
                wait = _loguniform(rng, lo, hi) if hi > lo else lo
            else:                      # range starting at 0: log-uniform from min(0.1, hi)
                wait = _loguniform(rng, min(0.1, hi), hi)
            dnp = bool(rng.random() < h['dnp_on_probability'])
            _run_for(model, wait, rf_on=False, dnp_on=dnp)
            events.append(dict(kind='wait', duration=wait, dnp=dnp))
    model.params.rf_enabled = False
    return model, events


# --------------------------------------------------------------------------- one sample
def design_from_model(model, mat: MaterialConfig, cfg: Dict, rng) -> Dict:
    """Optimize the RF program from the model's current state; return input and target fields."""
    model.params.dnp_enabled = bool(cfg['design']['dnp_during_pulse'])
    model.params.rf_enabled = False
    snapshot = PopulationSnapshot.capture(model)
    _, Ip, Im, total = model.spectrum_from_state(model.n)
    pol = model.polarizations()
    settings = OptimizerSettings.from_dict(mat.optimizer)
    for k, v in OPTIMIZER_PRESETS[cfg['design']['preset']].items(): setattr(settings, k, v)
    settings.max_total_power = cfg['design']['max_total_power']
    settings.max_total_exposure = cfg['design']['max_total_exposure']
    for k, v in cfg['design']['overrides'].items(): setattr(settings, k, v)
    settings.seed = int(rng.integers(2**31-1))
    t1 = time.time()
    result = optimize_tensor(snapshot, settings)
    rep = result.report
    _, fIp, fIm, ftotal = model.spectrum_from_state(result.final_n)
    post = cfg['post']; post_out = {}
    if post['steps'] > 0:
        pm = snapshot.make_model(); pm.n = np.array(result.final_n, copy=True)
        times = np.linspace(post['time']/post['steps'], post['time'], post['steps'])
        spectra, Ps, Qs = [], [], []; t_prev = 0.0
        for t in times:
            _run_for(pm, t-t_prev, rf_on=False, dnp_on=post['dnp']); t_prev = t
            spectra.append(pm.spectrum_from_state(pm.n)[3]); pol_t = pm.polarizations()
            Ps.append(pol_t['P']); Qs.append(pol_t['Q'])
        post_out = dict(post_times=times.astype(np.float32), post_spectra=np.array(spectra, dtype=np.float32),
                        post_P=np.array(Ps), post_Q=np.array(Qs))
    no_rf = rep.get('no_RF_at_same_endpoint')
    return dict(**post_out,
        state_P=pol['P'], state_Q=pol['Q'],
        spectrum=total.astype(np.float32), spectrum_plus=Ip.astype(np.float32), spectrum_minus=Im.astype(np.float32),
        populations=model.n.astype(np.float32),
        powers=np.asarray(result.powers, dtype=np.float32), duration=float(result.duration),
        final_P=rep['final']['P'], final_Q=rep['final']['Q'], delta_Q=rep['delta_Q'],
        no_rf_Q=no_rf.get('Q', float('nan')) if isinstance(no_rf, dict) else float('nan'),
        final_spectrum=ftotal.astype(np.float32),
        status=rep['status'], playback_dt=float(rep['recommended_playback_dt']),
        design_seconds=time.time()-t1)


def make_sample(sample_id: int, seed: int, cfg: Dict, materials: Dict[str, MaterialConfig]) -> Dict:
    """Random mode: random P0 and history, one optimized target."""
    rng = np.random.default_rng([seed, sample_id])
    name = cfg['materials'][int(rng.integers(len(cfg['materials'])))]
    mat = MaterialConfig.from_dict(materials[name].to_dict())
    p0 = float(rng.uniform(*cfg['p0_range']))
    mat.parameters.update(p0=p0, q0=None, rf_enabled=False, dnp_enabled=False)
    t0 = time.time()
    model, events = generate_state(mat, cfg['history'], rng)
    state_seconds = time.time()-t0
    out = design_from_model(model, mat, cfg, rng)
    return dict(out, sample_id=sample_id, material=name, p0=p0, events=events, group=sample_id,
                time_since_burn=float('nan'), state_seconds=state_seconds)


def make_sweep_group(group: int, seed: int, cfg: Dict, materials: Dict[str, MaterialConfig]) -> (List[Dict], List[Dict]):
    """Sweep mode: one (P0, history) pair evolved after the burn and designed at every time point.

    group = p0_index * histories_per_p0 + history_index. All time points of a group
    share one burn history, so they form a true evolution sequence. Samples get
    sample_id = group * n_times + time_index.
    """
    sw = cfg['sweep']; times = sorted(float(t) for t in sw['times'])
    M = int(sw['histories_per_p0'])
    p0 = float(sw['p0_values'][group//M])
    rng = np.random.default_rng([seed, 7919, group])
    name = cfg['materials'][int(rng.integers(len(cfg['materials'])))]
    mat = MaterialConfig.from_dict(materials[name].to_dict())
    mat.parameters.update(p0=p0, q0=None, rf_enabled=False, dnp_enabled=False)
    t0 = time.time()
    model, events = generate_state(mat, cfg['history'], rng)
    state_seconds = time.time()-t0
    samples, failures = [], []
    t_prev = 0.0
    for k, t in enumerate(times):
        _run_for(model, t-t_prev, rf_on=False, dnp_on=bool(sw['dnp'])); t_prev = t
        sid = group*len(times)+k
        try:
            work = snapshot_model(model)              # design on a copy; keep evolving the original
            out = design_from_model(work, mat, cfg, np.random.default_rng([seed, 7919, group, k]))
            samples.append(dict(out, sample_id=sid, material=name, p0=p0, events=events, group=group,
                                time_since_burn=t, state_seconds=state_seconds))
        except Exception as exc:
            failures.append(dict(sample_id=sid, group=group, time_since_burn=t,
                                 error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-2000:]))
    return samples, failures


def snapshot_model(model):
    return PopulationSnapshot.capture(model).make_model()


# --------------------------------------------------------------------------- shards
def shard_path(out: Path, start: int) -> Path:
    return out/f'shard_{start:08d}.npz'


def write_shard(path: Path, samples: List[Dict], store: Dict):
    arrays = dict(
        sample_id=np.array([s['sample_id'] for s in samples], dtype=np.int64),
        material=np.array([s['material'] for s in samples]),
        p0=np.array([s['p0'] for s in samples]),
        state_P=np.array([s['state_P'] for s in samples]), state_Q=np.array([s['state_Q'] for s in samples]),
        spectrum=np.stack([s['spectrum'] for s in samples]),
        powers=np.stack([s['powers'] for s in samples]),
        duration=np.array([s['duration'] for s in samples]),
        final_P=np.array([s['final_P'] for s in samples]), final_Q=np.array([s['final_Q'] for s in samples]),
        delta_Q=np.array([s['delta_Q'] for s in samples]), no_rf_Q=np.array([s['no_rf_Q'] for s in samples]),
        final_spectrum=np.stack([s['final_spectrum'] for s in samples]),
        status=np.array([s['status'] for s in samples]),
        playback_dt=np.array([s['playback_dt'] for s in samples]),
        state_seconds=np.array([s['state_seconds'] for s in samples]),
        design_seconds=np.array([s['design_seconds'] for s in samples]),
        events_json=np.array([json.dumps(s['events']) for s in samples]),
        group=np.array([s.get('group', s['sample_id']) for s in samples], dtype=np.int64),
        time_since_burn=np.array([s.get('time_since_burn', float('nan')) for s in samples]))
    if 'post_spectra' in samples[0]:
        for k in ('post_times', 'post_spectra', 'post_P', 'post_Q'):
            arrays[k] = np.stack([s[k] for s in samples])
    if store.get('branches', True):
        arrays['spectrum_plus'] = np.stack([s['spectrum_plus'] for s in samples])
        arrays['spectrum_minus'] = np.stack([s['spectrum_minus'] for s in samples])
    if store.get('populations', True):
        arrays['populations'] = np.stack([s['populations'] for s in samples])
    tmp = path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)   # atomic: a shard either exists completely or not at all


def run_block(args):
    """Worker entry point: generate samples [start, stop) and write one shard."""
    out, start, stop, seed, cfg = args
    out = Path(out); path = shard_path(out, start)
    if path.exists():
        return dict(start=start, written=0, skipped=True, failures=[])
    materials = {n: MaterialConfig.load(material_path(n)) for n in set(cfg['materials'])}
    samples, failures = [], []
    if cfg.get('sweep'):                       # in sweep mode a block is a range of groups
        for g in range(start, stop):
            s_, f_ = make_sweep_group(g, seed, cfg, materials)
            samples += s_; failures += f_
        if samples:
            write_shard(path, samples, cfg['store'])
        return dict(start=start, written=len(samples), skipped=False, failures=failures,
                    mean_design_seconds=float(np.mean([s['design_seconds'] for s in samples])) if samples else None)
    for i in range(start, stop):
        try:
            samples.append(make_sample(i, seed, cfg, materials))
        except Exception as exc:   # record and continue; deterministic seeds allow re-running one id
            failures.append(dict(sample_id=i, error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-2000:]))
    if samples:
        write_shard(path, samples, cfg['store'])
    return dict(start=start, written=len(samples), skipped=False, failures=failures,
                mean_design_seconds=float(np.mean([s['design_seconds'] for s in samples])) if samples else None)


def write_manifest(out: Path, cfg: Dict, seed: int, n_samples: int, shard_size: int):
    materials = {n: MaterialConfig.load(material_path(n)) for n in set(cfg['materials'])}
    grids = {n: (m.parameters['n_bins'], m.parameters['r_min'], m.parameters['r_max']) for n, m in materials.items()}
    if len(set(grids.values())) != 1:
        raise ValueError(f'All materials in one dataset must share a frequency grid; got {grids}')
    n, lo, hi = next(iter(grids.values()))
    manifest = dict(
        schema=SCHEMA, created=time.strftime('%Y-%m-%dT%H:%M:%S'), seed=seed, n_samples=n_samples,
        shard_size=shard_size, config=cfg, optimizer_preset=OPTIMIZER_PRESETS[cfg['design']['preset']],
        grid=dict(n_bins=n, r_min=lo, r_max=hi, R=np.linspace(lo, hi, n).tolist()),
        materials={k: v.to_dict() for k, v in materials.items()},
        fields={
            'spectrum': 'input: total signal I+ + I- at the manipulated/evolved state (n_bins)',
            'spectrum_plus/minus': 'input (optional): the two branch signals separately',
            'populations': 'input (optional): level populations (n_bins, 3), order +1, 0, -1',
            'state_P, state_Q': 'polarizations of the input state',
            'powers': 'target: per-bin RF command U_j of the optimal synchronized program (n_bins)',
            'duration': 'target: common pulse duration T (simulation time units)',
            'final_P, final_Q, delta_Q': 'outcome of the target program (predicted by the model)',
            'no_rf_Q': 'Q at the same endpoint with no RF (control)',
            'final_spectrum': 'total signal after the target program',
            'status': 'optimizer status (completed / needs_finer_time_step / ...)',
            'playback_dt': 'validated integration step for replaying the target',
            'events_json': 'the random history that produced the input state',
            'post_times, post_spectra, post_P, post_Q': 'optional: free evolution after the target pulse '
                                                       '(steps x n_bins spectra and their P, Q)',
            'material, p0': 'material name and starting Boltzmann P',
            'group': 'sweep mode: (P0, history) group index; samples of one group share one burn history',
            'time_since_burn': 'sweep mode: evolution time between the burn history and the input state (NaN in random mode)',
        })
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1))
    return manifest


def load_dataset(out: Path, fields: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Concatenate all shards (sorted by sample id)."""
    shards = sorted(Path(out).glob('shard_*.npz'))
    if not shards:
        raise FileNotFoundError(f'No shards in {out}')
    parts: Dict[str, list] = {}
    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            for k in (fields or z.files):
                parts.setdefault(k, []).append(z[k])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(data['sample_id']) if 'sample_id' in data else None
    return {k: v[order] for k, v in data.items()} if order is not None else data


def export_sample(out: Path, sample_id: int, program_path: Path, state_path: Optional[Path] = None):
    """Write a sample's target as a GUI-loadable program (and optionally its input state)."""
    manifest = json.loads((Path(out)/'manifest.json').read_text())
    data = load_dataset(out)
    idx = np.flatnonzero(data['sample_id'] == sample_id)
    if len(idx) == 0:
        raise KeyError(f'sample {sample_id} not found')
    i = int(idx[0]); R = np.array(manifest['grid']['R'])
    synchronized_program(R, data['powers'][i].astype(float), float(data['duration'][i])).save(program_path)
    if state_path is not None:
        if 'populations' not in data:
            raise ValueError('Dataset was generated without populations; cannot export the state')
        mat = MaterialConfig.from_dict(manifest['materials'][str(data['material'][i])])
        mat.parameters.update(p0=float(data['p0'][i]), q0=None)
        m = mat.make_model(preserve_state=False)
        m.n = data['populations'][i].astype(float)
        m.n *= m.mu[:, None]/m.n.sum(axis=1, keepdims=True)    # undo float32 rounding of packet mass
        PopulationSnapshot.capture(m).save(state_path)
    return i
