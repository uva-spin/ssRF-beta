"""Generate training data: manipulated/evolved spectra -> Q-optimizing RF programs.

  python3 generate_training_data.py generate --out data/run1 --samples 1000 --workers 8
  python3 generate_training_data.py generate --out data/run1 --samples 1000 --workers 8   # resumes
  python3 generate_training_data.py generate --out data/afp1 --samples 9 --afp-profile --p0-range 40 60
  python3 generate_training_data.py generate --out data/both --samples 100 --ssrf --afp-profile --p0-range 40 60 --dnp off --power-cap 1
  python3 generate_training_data.py info --data data/run1
  python3 generate_training_data.py combine --data data/run1
  python3 generate_training_data.py export --data data/run1 --sample 17 --program p17.json --state s17.npz

Every sample is a random history (RF burns, optional AFP, free evolution with or
without DNP) applied to a Boltzmann start, followed by the tensor optimizer run
from the resulting state. Output is sharded .npz files plus manifest.json.
See docs/COMMAND_LINE_DATA_GENERATION.md for the format and options.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # one BLAS thread per worker process
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ssrf_realtime import datagen


def physical_cpu_count() -> int:
    """Physical cores, not SMT threads. FFT Euler steps lose time on hyperthreads."""
    seen = set()
    root = Path('/sys/devices/system/cpu')
    if root.is_dir():
        for p in root.glob('cpu[0-9]*/topology/thread_siblings_list'):
            try:
                seen.add(p.read_text().strip())
            except OSError:
                pass
    return len(seen) or (os.cpu_count() or 2)


def default_workers() -> int:
    return max(1, physical_cpu_count() - 1)


def combine_shards(src: Path, dst: Path) -> int:
    data = datagen.load_dataset(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp.npz')
    np.savez_compressed(tmp, **data)
    tmp.replace(dst)
    return len(data['sample_id'])


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
    if a.afp_profile:
        if a.chains or a.p0_values:
            sys.exit('--afp-profile cannot be combined with --chains or --p0-values')
        cfg['design']['afp_profile'] = True
        if a.afp_prob is None:
            cfg['history']['afp_probability'] = 0.0
        if a.ssrf is None:
            cfg['design']['ssrf'] = False
    if a.ssrf is not None:
        cfg['design']['ssrf'] = bool(a.ssrf)
    if not cfg['design'].get('ssrf', True) and not cfg['design'].get('afp_profile'):
        sys.exit('need --ssrf and/or --afp-profile')
    if cfg['design'].get('afp_profile') and not cfg['design'].get('ssrf', True) and not a.n_burns:
        cfg['history']['n_burns'] = [0, 0]
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
    if a.chains:                                      # closed-loop chain mode
        ch = {'n_chains': a.chains, 'rounds': a.rounds,
              'dnp': (a.dnp == 'on') if a.dnp in ('on', 'off') else True}
        if a.p0_values: ch['p0_values'] = [v/100 if abs(v) > 1 else v for v in a.p0_values]
        if a.round_wait: ch['wait'] = list(a.round_wait)
        if a.round_wait_zero_prob is not None: ch['wait_zero_probability'] = a.round_wait_zero_prob
        if a.undershoot_power: ch['power_scale'] = list(a.undershoot_power)
        if a.undershoot_duration: ch['duration_scale'] = list(a.undershoot_duration)
        if a.profile_noise is not None: ch['profile_noise'] = a.profile_noise
        if a.exact_prob is not None: ch['exact_probability'] = a.exact_prob
        if not a.n_burns: cfg['history']['n_burns'] = [0, 1]
        cfg['chain'] = ch
    elif a.p0_values:                                 # sweep mode
        vals = [v/100 if abs(v) > 1 else v for v in a.p0_values]
        if not a.n_burns: cfg['history']['n_burns'] = [1, 1]
        cfg['history']['wait_zero_probability'] = 1.0   # evolution after the burn is set by --times
        cfg['history']['afp_probability'] = 0.0 if a.afp_prob is None else a.afp_prob
        cfg['sweep'] = {'p0_values': vals, 'histories_per_p0': a.histories_per_p0,
                        'times': sorted(a.times or [0.0]),
                        'dnp': (a.dnp == 'on') if a.dnp in ('on', 'off') else False}
    cfg = datagen.merged_config(cfg)
    sweep = cfg.get('sweep'); chain = cfg.get('chain')
    if chain:
        n_groups = int(chain['n_chains'])
        a.samples = n_groups*int(chain['rounds'])
    elif sweep:
        n_groups = len(sweep['p0_values'])*int(sweep['histories_per_p0'])
        a.samples = n_groups*len(sweep['times'])
    elif a.samples is None:
        sys.exit('--samples is required (or use --p0-values for sweep mode)')

    n_modes = len(datagen.target_modes(cfg))
    n_events = a.samples
    n_rows = n_events * n_modes if not (sweep or chain) else a.samples
    if a.shard_size is None:
        a.shard_size = 1 if (sweep or chain) else n_modes   # one input event per shard

    manifest_file = out/'manifest.json'
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text())
        if old['config'] != cfg or old['seed'] != a.seed or old['shard_size'] != a.shard_size:
            sys.exit(f'{out} already holds a dataset with a different config/seed/shard size. '
                     'Use a new --out directory, or the same options to resume.')
        n_total = max(n_rows, old['n_samples'])
    else:
        n_total = n_rows
    datagen.write_manifest(out, cfg, a.seed, n_total, a.shard_size)

    if sweep or chain:   # one shard per `shard_size` groups / chains
        n_units, unit = n_groups, ('chain' if chain else 'group')
    else:
        n_units, unit = n_total, 'sample'
    blocks = [(str(out), s, min(s+a.shard_size, n_units), a.seed, cfg)
              for s in range(0, n_units, a.shard_size)
              if not datagen.shard_path(out, s).exists()]
    extra = (f' Sweep: P0 {sweep["p0_values"]}, {sweep["histories_per_p0"]} histories each, '
             f'times {sweep["times"]} -> {n_groups} groups.') if sweep else (
             f' Chains: {n_groups} x {chain["rounds"]} rounds (optimize -> apply imperfectly -> evolve).' if chain else '')
    if n_modes > 1 and not sweep and not chain:
        extra = (f' {n_events} events × {n_modes} manipulations '
                 f'({", ".join(datagen.target_modes(cfg))}).') + extra
    n_workers = min(a.workers, len(blocks)) if blocks else 0
    print(f'{n_total} samples in shards of {a.shard_size} {unit}(s): {len(blocks)} shard(s) to do, '
          f'{n_workers} worker(s), preset {cfg["design"]["preset"]}.{extra}', flush=True)
    t0 = time.time(); done = 0; failures = []; finished = 0
    fail_log = out/'failures.jsonl'
    if blocks:
        ctx = get_context('spawn')
        with ctx.Pool(n_workers, initializer=datagen.init_worker, initargs=(cfg,)) as pool:
            for r in pool.imap_unordered(datagen.run_block, blocks):
                done += r['written']; failures += r['failures']; finished += 1
                for f in r['failures']:
                    with fail_log.open('a') as fh: fh.write(json.dumps(f)+'\n')
                elapsed = max(time.time()-t0, 1e-9)
                eta = (len(blocks)-finished) * elapsed / finished
                msg = f'shard {r["start"]:>8d}: {r["written"]} written'
                if r.get('mean_design_seconds'): msg += f', {r["mean_design_seconds"]:.1f} s/design'
                print(f'{msg} | {finished}/{len(blocks)} shards, {done} new, '
                      f'{len(failures)} fail, {3600*done/elapsed:.0f} samples/h, ~{eta:.0f}s left', flush=True)
    print(f'Done in {time.time()-t0:.0f} s. Failures logged to {fail_log}' if failures else f'Done in {time.time()-t0:.0f} s.')
    if list(out.glob('shard_*.npz')):
        n = combine_shards(out, out.with_suffix('.npz'))
        print(f'combined {n} samples -> {out.with_suffix(".npz")}')
    plot_examples(out)


def plot_examples(out: Path):
    """Write lineshape + power-profile plots for a few samples (min/median/max ΔQ)."""
    out = Path(out)
    if not (out/'manifest.json').exists() or not list(out.glob('shard_*.npz')):
        return
    man = json.loads((out/'manifest.json').read_text())
    R = np.asarray(man['grid']['R'], dtype=float)
    d = datagen.load_dataset(out, ['sample_id', 'spectrum', 'final_spectrum', 'powers',
                                   'delta_Q', 'state_Q', 'final_Q', 'duration'])
    n = len(d['sample_id'])
    if n == 0:
        return
    finite = np.flatnonzero(np.isfinite(d['delta_Q']))
    order = finite[np.argsort(d['delta_Q'][finite])] if len(finite) else np.arange(n)
    picks = [int(order[0])]
    if len(order) > 2:
        picks.append(int(order[len(order)//2]))
    if len(order) > 1:
        picks.append(int(order[-1]))
    picks = list(dict.fromkeys(picks))
    names = {picks[0]: 'smallest ΔQ', picks[-1]: 'largest ΔQ'}
    if len(picks) == 3:
        names[picks[1]] = 'median ΔQ'
    elif len(picks) == 1:
        names[picks[0]] = 'example'
    fig, ax = plt.subplots(len(picks), 2, figsize=(9.4, 2.8*len(picks)), sharex='col', squeeze=False)
    for row, i in enumerate(picks):
        sid = int(d['sample_id'][i])
        ax[row, 0].plot(R, d['spectrum'][i], color='#3b6ea5', label='input I+ + I−')
        if np.isfinite(d['final_spectrum'][i]).any():
            ax[row, 0].plot(R, d['final_spectrum'][i], color='#c45c26', label='after the pulse')
        ax[row, 0].set_ylabel('intensity')
        ax[row, 0].set_title(
            f'{names[i]}: sample {sid}, Q {d["state_Q"][i]:.3f} → {d["final_Q"][i]:.3f}'
            f' (ΔQ {d["delta_Q"][i]:+.3f}), T = {d["duration"][i]:.3g}')
        ax[row, 1].step(R, d['powers'][i], where='mid', color='#2f6f4e')
        ax[row, 1].set_ylabel('RF command U')
        ax[row, 1].set_title(f'optimized profile, total power {d["powers"][i].sum():.3g} U')
        if row == 0:
            ax[row, 0].legend(frameon=False, loc='upper right')
        for a in ax[row]:
            a.grid(True, alpha=0.3)
            a.set_axisbelow(True)
    ax[-1, 0].set_xlabel('R')
    ax[-1, 1].set_xlabel('R')
    fig.tight_layout()
    dest = out/'example.png'
    fig.savefig(dest, dpi=160)
    plt.close(fig)
    print(f'Wrote {dest} ({len(picks)} sample{"s" if len(picks)!=1 else ""})')


def cmd_info(a):
    out = Path(a.data); m = json.loads((out/'manifest.json').read_text())
    d = datagen.load_dataset(out, ['sample_id', 'state_Q', 'final_Q', 'delta_Q', 'duration', 'status',
                                   'manipulation', 'design_seconds', 'material'])
    n = len(d['sample_id'])
    print(f"{out}: {n}/{m['n_samples']} samples, grid {m['grid']['n_bins']} bins, preset {m['config']['design']['preset']}, "
          f"materials {m['config']['materials']}")
    if n:
        st, cnt = np.unique(d['status'], return_counts=True)
        print('status:', dict(zip(st.tolist(), cnt.tolist())))
        if 'manipulation' in d:
            mt, mc = np.unique(d['manipulation'], return_counts=True)
            print('manipulation:', dict(zip(mt.tolist(), mc.tolist())))
        for k in ('state_Q', 'final_Q', 'delta_Q', 'duration', 'design_seconds'):
            v = d[k][np.isfinite(d[k])]
            if len(v): print(f'  {k:15s} mean {v.mean():.4g}  min {v.min():.4g}  max {v.max():.4g}')
    fl = out/'failures.jsonl'
    if fl.exists(): print('failures:', sum(1 for _ in fl.open()))


def cmd_combine(a):
    src = Path(a.data)
    dst = Path(a.out) if a.out else src.with_suffix('.npz')
    n = combine_shards(src, dst)
    print(f'{n} samples -> {dst}')
    plot_examples(src)


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
    g.add_argument('--samples', type=int,
                   help='input events (random mode); with --ssrf --afp-profile each event writes two rows')
    g.add_argument('--workers', type=int, default=default_workers(),
                   help=f'parallel processes (default {default_workers()}: physical cores minus one)')
    g.add_argument('--shard-size', type=int, help='rows per shard (random mode, default one input event) or '
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
    g.add_argument('--ssrf', action=argparse.BooleanOptionalAction, default=None,
                   help='write an optimized RF row per input event (default on, unless --afp-profile is used alone)')
    g.add_argument('--afp-profile', action='store_true',
                   help='write a Q<0 AFP row per input event (spectrum immediately after the flip). '
                        'With --ssrf, both rows share the same history and go in one dataset.')
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
                   help='initial polarizations, percent or fraction (e.g. 25 27 29). Enables sweep mode '
                        '(with --chains: P0 values cycled over the chains).')
    w.add_argument('--histories-per-p0', type=int, default=8, help='random burn histories per P0 (default 8)')
    w.add_argument('--times', type=float, nargs='+', metavar='T',
                   help='evolution times after the burn at which inputs are taken (default 0)')
    c = g.add_argument_group('chain mode (closed loop: optimize -> apply an undershot program -> evolve -> optimize again)')
    c.add_argument('--chains', type=int, help='number of chains; enables chain mode (samples = chains x rounds)')
    c.add_argument('--rounds', type=int, default=4, help='optimizations per chain (default 4)')
    c.add_argument('--round-wait', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='evolution after each applied pulse before the next input (default 2 40, log-uniform)')
    c.add_argument('--round-wait-zero-prob', type=float, help='probability of no wait (default 0)')
    c.add_argument('--undershoot-power', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='applied/optimal total power, uniform (default 0.3 1.0)')
    c.add_argument('--undershoot-duration', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='applied/optimal duration, uniform (default 0.2 1.0)')
    c.add_argument('--profile-noise', type=float, help='log-normal per-bin noise on the applied profile (default 0.5)')
    c.add_argument('--exact-prob', type=float, help='probability the optimal program is applied exactly (default 0.15)')
    q = g.add_argument_group('output evolution (after the optimized pulse)')
    q.add_argument('--post-time', type=float, help='evolve this long after the optimized pulse (simulation time units)')
    q.add_argument('--post-steps', type=int, help='number of evenly spaced spectra recorded over --post-time')
    q.add_argument('--post-dnp', action='store_true', help='DNP on during the post-pulse evolution')
    g.set_defaults(func=cmd_generate)
    i = sub.add_parser('info', help='summarize a dataset'); i.add_argument('--data', required=True); i.set_defaults(func=cmd_info)
    c = sub.add_parser('combine', help='concatenate shard_*.npz in a dataset directory into one file')
    c.add_argument('--data', required=True, help='dataset directory (reads shard_*.npz)')
    c.add_argument('--out', help='output .npz (default: <directory>.npz next to that directory)')
    c.set_defaults(func=cmd_combine)
    e = sub.add_parser('export', help='write one sample as a GUI program (+ state)')
    e.add_argument('--data', required=True); e.add_argument('--sample', type=int, required=True)
    e.add_argument('--program', required=True); e.add_argument('--state')
    e.set_defaults(func=cmd_export)
    a = ap.parse_args(); a.func(a)


if __name__ == '__main__':
    main()
