"""Train a network that reads a spectrum and outputs the vector and tensor polarization.

  python3 generate_training_data.py generate --out data/pq --samples 20000 --states-only --p0-range 10 60
  python3 -m ml.train_pq --data data/pq --out ml_runs/pq --epochs 200 --noise 0.002

Input:  the total spectrum (and its mirror R -> -R), on the dataset's grid.
Output: P and Q, plus (auxiliary) the I+ branch signal, from which I- = total - I+.
        Learning the branch split is what makes Q recoverable: Q = A+ - A- (branch areas),
        P = A+ + A-.

Evaluation on held-out samples, separately for unmanipulated (Boltzmann) and manipulated
states, against two classical baselines:
  - area method:       P from the total area (linear calibration on the training set);
  - Boltzmann relation: Q = 2 - sqrt(4 - 3P^2) from the area-method P. This is exact only
                        for unmanipulated states.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.data import split_groups
from ml.model import ConvBlock
from ssrf_realtime.datagen import load_dataset


class SpectrumToPQ(nn.Module):
    """1D U-Net trunk: per-bin head for the I+ branch, pooled head for (P, Q)."""
    def __init__(self, width=32, in_channels=2):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.enc3 = ConvBlock(in_channels, w), ConvBlock(w, 2*w), ConvBlock(2*w, 4*w)
        self.dec2, self.dec1 = ConvBlock(6*w, 2*w), ConvBlock(3*w, w)
        self.branch = nn.Conv1d(w, 1, 1)
        self.pq = nn.Sequential(nn.Linear(8*w, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(F.max_pool1d(e1, 2)); e3 = self.enc3(F.max_pool1d(e2, 2))
        pq = self.pq(torch.cat([e3.mean(-1), e3.amax(-1)], 1))
        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[-1], mode='linear'), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-1], mode='linear'), e1], 1))
        return pq, self.branch(d1).squeeze(1)


def unmanipulated_mask(events_json):
    """True where the history had no RF burn and no AFP (pure Boltzmann + evolution)."""
    return np.array([not any(e['kind'] in ('burn', 'afp') for e in json.loads(s)) for s in events_json])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=200); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--noise', type=float, default=0.0, help='Gaussian noise added to inputs, relative to the max |signal|')
    ap.add_argument('--branch-weight', type=float, default=1.0, help='weight of the auxiliary I+ branch loss')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = load_dataset(a.data, ['sample_id', 'group', 'spectrum', 'spectrum_plus', 'state_P', 'state_Q', 'events_json'])
    groups = d['group'] if 'group' in d else d['sample_id']
    tr, va, te = split_groups(groups, a.seed)
    # Absolute scale matters (P is proportional to area), so use ONE global scale, not per-sample.
    scale = float(np.abs(d['spectrum'][tr]).max())
    X = d['spectrum']/scale; B = d['spectrum_plus']/scale
    Y = np.c_[d['state_P'], d['state_Q']].astype(np.float32)
    feats = lambda x: torch.from_numpy(np.stack([x, x[:, ::-1]], 1).astype(np.float32).copy())
    dev = torch.device(a.device)
    model = SpectrumToPQ(a.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    Xtr, Btr, Ytr = torch.from_numpy(X[tr].astype(np.float32)), torch.from_numpy(B[tr].astype(np.float32)), torch.from_numpy(Y[tr])
    best, t0 = np.inf, time.time()
    ref = va if len(va) else tr
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(perm), a.batch):
            b = perm[i:i+a.batch]; x = Xtr[b]
            if a.noise > 0:
                x = x+a.noise*torch.randn_like(x)
            xin = torch.stack([x, torch.flip(x, [1])], 1).to(dev)
            pq, br = model(xin)
            loss = F.mse_loss(pq, Ytr[b].to(dev))+a.branch_weight*F.mse_loss(br, Btr[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            pq, _ = model(feats(X[ref]).to(dev))
            vl = F.mse_loss(pq.cpu(), torch.from_numpy(Y[ref])).item()
        if vl < best:
            best = vl; torch.save(model.state_dict(), out/'model_pq.pt')
        if ep % 25 == 0 or ep == a.epochs-1:
            print(f'epoch {ep:4d}  val MSE(P,Q) {vl:.3e}  {time.time()-t0:.0f}s', flush=True)
    model.load_state_dict(torch.load(out/'model_pq.pt', map_location=dev)); model.eval()

    if len(te) == 0:
        print('No test samples.'); return
    with torch.no_grad():
        xt = X[te]
        if a.noise > 0:
            xt = xt+a.noise*np.random.default_rng(1).standard_normal(xt.shape)
        pq, _ = model(feats(xt).to(dev))
    pq = pq.cpu().numpy()
    # Classical baselines: area calibration for P, Boltzmann relation for Q.
    area = lambda x: x.sum(1)
    k = np.polyfit(area(X[tr]), Y[tr, 0], 1)
    P_area = np.polyval(k, area(xt)); Q_boltz = 2-np.sqrt(np.clip(4-3*P_area**2, 0, None))
    unm = unmanipulated_mask(d['events_json'][te])
    def mae(pred, true, m): return float(np.mean(np.abs(pred[m]-true[m]))) if m.any() else None
    report = {}
    for name, m in (('all', np.ones(len(te), bool)), ('unmanipulated', unm), ('manipulated', ~unm)):
        report[name] = dict(n=int(m.sum()),
                            network=dict(P_MAE=mae(pq[:, 0], Y[te, 0], m), Q_MAE=mae(pq[:, 1], Y[te, 1], m)),
                            area_and_boltzmann=dict(P_MAE=mae(P_area, Y[te, 0], m), Q_MAE=mae(Q_boltz, Y[te, 1], m)))
    (out/'report_pq.json').write_text(json.dumps(dict(args=vars(a), scale=scale, best_val=best, test=report), indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
