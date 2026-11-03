"""Generate training data: manipulated/evolved spectra -> Q-optimizing RF programs.

  python3 generate_training_data.py generate --out data/run1 --samples 1000 --workers 8
  python3 generate_training_data.py generate --out data/run1 --samples 1000 --workers 8   # resumes
  python3 generate_training_data.py info --data data/run1
  python3 generate_training_data.py export --data data/run1 --sample 17 --program p17.json --state s17.npz

Every sample is a random history (RF burns, optional AFP, free evolution with or
without DNP) applied to a Boltzmann start, followed by the tensor optimizer run
from the resulting state. Output is sharded .npz files plus manifest.json.
See docs/COMMAND_LINE_DATA_GENERATION.md for the format and options.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # one BLAS thread per worker process
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ssrf_realtime import datagen


def cmd_generate(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    user = json.loads(Path(a.config).read_text()) if a.config else {}
    cfg = datagen.merged_config(user)
    if a.materials: cfg['materials'] = a.materials.split(',')
    if a.preset: cfg['design']['preset'] = a.preset
    if a.dnp in ('on', 'off'):                       # one switch for the whole sample
        on = a.dnp == 'on'
        cfg['history']['dnp_on_probability'] = 1.0 if on else 0.0
        cfg['history']['dnp_during_burns'] = on
        cfg['design']['dnp_during_pulse'] = on
        cfg['post']['dnp'] = on
    if a.power_cap is not None: cfg['design']['max_total_power'] = a.power_cap
    if a.exposure_cap is not None: cfg['design']['max_total_exposure'] = a.exposure_cap
    if a.design_dnp: cfg['design']['dnp_during_pulse'] = True
    if a.no_populations: cfg['store']['populations'] = False
    if a.states_only: cfg['design']['skip'] = True
    if a.p0_range:
        lo, hi = a.p0_range
        if max(abs(lo), abs(hi)) > 1: lo, hi = lo/100, hi/100      # accept percent
        cfg['p0_range'] = [lo, hi]
    h = cfg['history']
    for opt, key in (('n_burns', 'n_burns'), ('burn_duration', 'burn_duration'), ('burn_power', 'burn_total_power_U'),
                     ('burn_width', 'burn_width_R'), ('wait_range', 'wait')):
        if getattr(a, opt): h[key] = list(getattr(a, opt))
    if a.n_burns: h['n_burns'] = [int(x) for x in a.n_burns]
    if a.wait_zero_prob is not None: h['wait_zero_probability'] = a.wait_zero_prob
    if a.afp_prob is not None: h['afp_probability'] = a.afp_prob
    if a.dnp_prob is not None: h['dnp_on_probability'] = a.dnp_prob
    if a.post_time is not None or a.post_steps is not None:
        cfg['post']['time'] = a.post_time or 0.0
        cfg['post']['steps'] = a.post_steps or 0
    if a.post_dnp: cfg['post']['dnp'] = True
    if a.p0_values:                                   # sweep mode
        vals = [v/100 if abs(v) > 1 else v for v in a.p0_values]
        if not a.n_burns: cfg['history']['n_burns'] = [1, 1]
        cfg['history']['wait_zero_probability'] = 1.0   # evolution after the burn is set by --times
        cfg['history']['afp_probability'] = 0.0 if a.afp_prob is None else a.afp_prob
        cfg['sweep'] = {'p0_values': vals, 'histories_per_p0': a.histories_per_p0,
                        'times': sorted(a.times or [0.0]),
                        'dnp': (a.dnp == 'on') if a.dnp in ('on', 'off') else False}
    cfg = datagen.merged_config(cfg)
    sweep = cfg.get('sweep')
    if a.shard_size is None: a.shard_size = 1 if sweep else 16
    if sweep:
        n_groups = len(sweep['p0_values'])*int(sweep['histories_per_p0'])
        a.samples = n_groups*len(sweep['times'])
    elif a.samples is None:
        sys.exit('--samples is required (or use --p0-values for sweep mode)')

    manifest_file = out/'manifest.json'
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text())
        if old['config'] != cfg or old['seed'] != a.seed or old['shard_size'] != a.shard_size:
            sys.exit(f'{out} already holds a dataset with a different config/seed/shard size. '
                     'Use a new --out directory, or the same options to resume.')
        n_total = max(a.samples, old['n_samples'])
    else:
        n_total = a.samples
    datagen.write_manifest(out, cfg, a.seed, n_total, a.shard_size)

    if sweep:   # one shard per group of `shard_size` (P0, history) pairs
        n_units, unit = n_groups, 'group'
    else:
        n_units, unit = n_total, 'sample'
    blocks = [(str(out), s, min(s+a.shard_size, n_units), a.seed, cfg)
              for s in range(0, n_units, a.shard_size)
              if not datagen.shard_path(out, s).exists()]
    extra = (f' Sweep: P0 {sweep["p0_values"]}, {sweep["histories_per_p0"]} histories each, '
             f'times {sweep["times"]} -> {n_groups} groups.') if sweep else ''
    print(f'{n_total} samples in shards of {a.shard_size} {unit}(s): {len(blocks)} shard(s) to do, '
          f'{a.workers} worker(s), preset {cfg["design"]["preset"]}.{extra}', flush=True)
    t0 = time.time(); done = 0; failures = []
    fail_log = out/'failures.jsonl'
    ctx = get_context('spawn')
    with ctx.Pool(a.workers) as pool:
        for r in pool.imap_unordered(datagen.run_block, blocks):
            done += r['written']; failures += r['failures']
            for f in r['failures']:
                with fail_log.open('a') as fh: fh.write(json.dumps(f)+'\n')
            rate = done/max(time.time()-t0, 1e-9)
            msg = f'shard {r["start"]:>8d}: {r["written"]} written'
            if r.get('mean_design_seconds'): msg += f', {r["mean_design_seconds"]:.1f} s/design'
            print(f'{msg} | total {done} new samples, {len(failures)} failures, {3600*rate:.0f} samples/h', flush=True)
    print(f'Done in {time.time()-t0:.0f} s. Failures logged to {fail_log}' if failures else f'Done in {time.time()-t0:.0f} s.')


def cmd_info(a):
    out = Path(a.data); m = json.loads((out/'manifest.json').read_text())
    d = datagen.load_dataset(out, ['sample_id', 'state_Q', 'final_Q', 'delta_Q', 'duration', 'status', 'design_seconds', 'material'])
    n = len(d['sample_id'])
    print(f"{out}: {n}/{m['n_samples']} samples, grid {m['grid']['n_bins']} bins, preset {m['config']['design']['preset']}, "
          f"materials {m['config']['materials']}")
    if n:
        st, cnt = np.unique(d['status'], return_counts=True)
        print('status:', dict(zip(st.tolist(), cnt.tolist())))
        for k in ('state_Q', 'final_Q', 'delta_Q', 'duration', 'design_seconds'):
            v = d[k][np.isfinite(d[k])]
            if len(v): print(f'  {k:15s} mean {v.mean():.4g}  min {v.min():.4g}  max {v.max():.4g}')
    fl = out/'failures.jsonl'
    if fl.exists(): print('failures:', sum(1 for _ in fl.open()))


def cmd_export(a):
    i = datagen.export_sample(Path(a.data), a.sample, Path(a.program), Path(a.state) if a.state else None)
    m = json.loads((Path(a.data)/'manifest.json').read_text())
    d = datagen.load_dataset(Path(a.data), ['sample_id', 'material', 'final_Q', 'duration', 'playback_dt'])
    print(f'sample {a.sample}: material {d["material"][i]}, target duration {d["duration"][i]:.4g}, '
          f'predicted Q {d["final_Q"][i]:.4f}. Wrote {a.program}' + (f' and {a.state}' if a.state else ''))
    print(f'Open it with: python3 run_app.py --state {a.state} --program {a.program}' if a.state else
          f'Open it with: python3 run_app.py {d["material"][i]} --program {a.program}')
    if d['playback_dt'][i] < 1e-3:
        print(f'Note: replay needs integration step {d["playback_dt"][i]:.3g}; set dt accordingly for an exact replay.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('generate', help='generate (or resume) a dataset')
    g.add_argument('--out', required=True, help='output directory')
    g.add_argument('--samples', type=int, help='total number of samples (random mode)')
    g.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)-1))
    g.add_argument('--shard-size', type=int, help='samples per shard (random mode, default 16) or '
                   '(P0, history) groups per shard (sweep mode, default 1)')
    g.add_argument('--seed', type=int, default=12345)
    g.add_argument('--materials', help='comma-separated material names (default d_butanol_matched)')
    g.add_argument('--preset', choices=sorted(datagen.OPTIMIZER_PRESETS), help='optimizer effort (default fast)')
    g.add_argument('--power-cap', type=float, help='total RF power limit in U (1 = 0.1 W); default none')
    g.add_argument('--exposure-cap', type=float, help='total exposure limit in U x time (34 = 3.4 J); default none')
    g.add_argument('--dnp', choices=['on', 'off', 'random'], default='random',
                   help='DNP for the whole sample: on (history, optimized pulse and post-pulse evolution), '
                        'off (none), or random (default: random during history waits, off during the pulse). '
                        'The flags below override individual parts.')
    g.add_argument('--design-dnp', action='store_true', help='optimize with DNP on during the pulse')
    g.add_argument('--no-populations', action='store_true', help='do not store level populations')
    g.add_argument('--states-only', action='store_true',
                   help='skip the optimizer: store spectra with their P and Q labels only (fast; for polarization models)')
    g.add_argument('--config', help='JSON file overriding datagen.DEFAULT_CONFIG (history ranges etc.)')
    h = g.add_argument_group('input states (random history before the optimized pulse)')
    h.add_argument('--p0-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='initial vector polarization range, fraction or percent (e.g. 20 60); default 0.30 0.60')
    h.add_argument('--n-burns', type=float, nargs=2, metavar=('MIN', 'MAX'), help='number of history RF burns (default 0 3)')
    h.add_argument('--burn-duration', type=float, nargs=2, metavar=('MIN', 'MAX'), help='burn duration range, log-uniform (default 0.01 20)')
    h.add_argument('--burn-power', type=float, nargs=2, metavar=('MIN', 'MAX'), help='total burn power range in U, log-uniform (default 0.01 5)')
    h.add_argument('--burn-width', type=float, nargs=2, metavar=('MIN', 'MAX'), help='burn window width in R, log-uniform (default 0.02 0.6)')
    h.add_argument('--wait-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='free evolution after each history pulse before the input is recorded (default 0 600)')
    h.add_argument('--wait-zero-prob', type=float, help='probability of no evolution after a pulse (default 0.2)')
    h.add_argument('--afp-prob', type=float, help='probability of an AFP window in the history (default 0.2)')
    h.add_argument('--dnp-prob', type=float, help='probability DNP is on during each evolution (default 0.3)')
    w = g.add_argument_group('sweep mode (structured: P0 grid x burn histories x evolution times)')
    w.add_argument('--p0-values', type=float, nargs='+', metavar='P',
                   help='initial polarizations, percent or fraction (e.g. 25 27 29). Enables sweep mode.')
    w.add_argument('--histories-per-p0', type=int, default=8, help='random burn histories per P0 (default 8)')
    w.add_argument('--times', type=float, nargs='+', metavar='T',
                   help='evolution times after the burn at which inputs are taken (default 0)')
    q = g.add_argument_group('output evolution (after the optimized pulse)')
    q.add_argument('--post-time', type=float, help='evolve this long after the optimized pulse (simulation time units)')
    q.add_argument('--post-steps', type=int, help='number of evenly spaced spectra recorded over --post-time')
    q.add_argument('--post-dnp', action='store_true', help='DNP on during the post-pulse evolution')
    g.set_defaults(func=cmd_generate)
    i = sub.add_parser('info', help='summarize a dataset'); i.add_argument('--data', required=True); i.set_defaults(func=cmd_info)
    e = sub.add_parser('export', help='write one sample as a GUI program (+ state)')
    e.add_argument('--data', required=True); e.add_argument('--sample', type=int, required=True)
    e.add_argument('--program', required=True); e.add_argument('--state')
    e.set_defaults(func=cmd_export)
    a = ap.parse_args(); a.func(a)


if __name__ == '__main__':
    main()
