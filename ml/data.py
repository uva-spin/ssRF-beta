"""Load generator shards into tensors with a deterministic train/val/test split."""
import json
from pathlib import Path

import numpy as np
import torch

from ssrf_realtime.datagen import load_dataset

FIELDS = ['sample_id', 'spectrum', 'powers', 'duration', 'state_P', 'state_Q', 'final_Q', 'no_rf_Q',
          'status', 'populations', 'material', 'p0', 'group', 'time_since_burn']


def split_groups(groups, seed=0, fractions=(0.8, 0.1, 0.1), test_mask=None):
    """Split by group (all time points of one burn history stay together) to avoid leakage.

    test_mask: optional boolean array forcing samples into the test set (e.g. held-out P0).
    """
    rng = np.random.default_rng(seed)
    forced = np.zeros(len(groups), bool) if test_mask is None else np.asarray(test_mask, bool)
    free = np.unique(groups[~forced]); rng.shuffle(free)
    n_tr = int(round(fractions[0]*len(free)/(fractions[0]+fractions[1]+(0 if test_mask is not None else fractions[2]))))
    if test_mask is None:
        n_va = int(round(fractions[1]*len(free)))
        tr_g, va_g, te_g = free[:n_tr], free[n_tr:n_tr+n_va], free[n_tr+n_va:]
    else:
        tr_g, va_g, te_g = free[:n_tr], free[n_tr:], np.unique(groups[forced])
    pick = lambda gs: np.flatnonzero(np.isin(groups, gs) & (forced if gs is te_g and test_mask is not None else True))
    return pick(tr_g), pick(va_g), pick(te_g)


class PulseData:
    def __init__(self, data_dir, seed=0, completed_only=True, holdout_p0=None, test_dir=None):
        """test_dir: optional independent dataset (same grid/material) used only for testing;
        data_dir is then split into train and validation only."""
        self.dir = Path(data_dir)
        self.manifest = json.loads((self.dir/'manifest.json').read_text())
        d = load_dataset(self.dir, FIELDS)
        d['from_test_dir'] = np.zeros(len(d['sample_id']), bool)
        if test_dir:
            e = load_dataset(Path(test_dir), FIELDS)
            if json.loads((Path(test_dir)/'manifest.json').read_text())['grid']['n_bins'] != self.manifest['grid']['n_bins']:
                raise ValueError('test dataset has a different frequency grid')
            e['group'] = e['group']+int(d['group'].max())+1          # keep groups distinct
            e['from_test_dir'] = np.ones(len(e['sample_id']), bool)
            d = {k: np.concatenate([d[k], e[k]]) for k in d}
        keep = np.ones(len(d['sample_id']), bool)
        if completed_only:
            keep &= d['status'] == 'completed'
        keep &= d['powers'].sum(1) > 0                         # targets with RF
        self.raw = {k: v[keep] for k, v in d.items()}
        groups = self.raw['group'] if 'group' in self.raw else self.raw['sample_id']
        mask = self.raw['from_test_dir'].copy() if test_dir else None
        if holdout_p0:
            mask = np.zeros(len(groups), bool) if mask is None else mask
            for p in holdout_p0:
                mask |= np.isclose(self.raw['p0'], p, atol=1e-6)
        self.train, self.val, self.test = split_groups(groups, seed, test_mask=mask)
        t = self.raw.get('time_since_burn')
        self.use_time = t is not None and np.all(np.isfinite(t)) and np.ptp(t) > 0
        self.t_scale = float(np.max(t)) if self.use_time else 1.0
        x = self.raw['spectrum'].astype(np.float32)
        tot = self.raw['powers'].sum(1)
        self.x_scale = float(np.abs(x[self.train]).max())
        self.total_mu, self.total_sd = float(tot[self.train].mean()), float(tot[self.train].std()+1e-6)
        lt = np.log(self.raw['duration'])
        self.logT_mu, self.logT_sd = float(lt[self.train].mean()), float(lt[self.train].std()+1e-6)

    def norm(self):
        return dict(x_scale=self.x_scale, total_mu=self.total_mu, total_sd=self.total_sd,
                    logT_mu=self.logT_mu, logT_sd=self.logT_sd, use_time=self.use_time, t_scale=self.t_scale)

    @property
    def in_channels(self):
        return 3 if self.use_time else 2

    @staticmethod
    def features(spectrum, x_scale):
        x = np.asarray(spectrum, np.float32)/x_scale
        return np.stack([x, x[..., ::-1]], axis=-2)            # (.., 2, N): spectrum and mirror

    def tensors(self, idx):
        r = self.raw
        f = self.features(r['spectrum'][idx], self.x_scale)
        if self.use_time:                                       # constant channel: time since burn
            tc = (r['time_since_burn'][idx]/self.t_scale).astype(np.float32)
            f = np.concatenate([f, np.broadcast_to(tc[:, None, None], (len(idx), 1, f.shape[-1]))], axis=1)
        x = torch.from_numpy(np.ascontiguousarray(f))
        p = r['powers'][idx].astype(np.float32); tot = p.sum(1)
        prof = torch.from_numpy(p/tot[:, None])
        total = torch.from_numpy(((tot-self.total_mu)/self.total_sd).astype(np.float32))
        logT = torch.from_numpy(((np.log(r['duration'][idx])-self.logT_mu)/self.logT_sd).astype(np.float32))
        return x, prof, total, logT
