"""Train the spectrum -> RF-program network and evaluate it with the physics model.

  python3 -m ml.train --data data/pilot_dnp_on --out ml_runs/pilot --epochs 300

Loss: KL divergence between the target and predicted frequency profiles of RF power,
plus mean-squared error on the standardized total power and log duration. After
training, each test state is run through the simulator with
  (a) the optimizer's program (the target),
  (b) the network's program,
  (c) baselines: the mean training program, and nearest-neighbour lookup of the
      training sample with the closest spectrum.
The report gives Q and the fraction of the optimizer's gain over no RF achieved.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.data import PulseData
from ml.model import SpectrumToPulse
from ml.physics_eval import PhysicsEvaluator


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def loss_fn(logits, total_hat, logT_hat, prof, total, logT, w_scalar=0.5):
    logq = F.log_softmax(logits, dim=1)
    kl = (prof*(torch.log(prof.clamp_min(1e-12))-logq)).sum(1).mean()
    return kl+w_scalar*(F.mse_loss(total_hat, total)+F.mse_loss(logT_hat, logT)), kl


def predict(model, ds, idx, device):
    model.eval()
    x, *_ = ds.tensors(idx)
    with torch.no_grad():
        logits, t_hat, lT_hat = model(x.to(device))
    prof = torch.softmax(logits, 1).cpu().numpy()
    total = t_hat.cpu().numpy()*ds.total_sd+ds.total_mu
    T = np.exp(lT_hat.cpu().numpy()*ds.logT_sd+ds.logT_mu)
    return prof*np.clip(total, 1e-6, None)[:, None], T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='auto')
    ap.add_argument('--eval-max', type=int, default=200, help='max test states for the physics evaluation')
    ap.add_argument('--holdout-p0', type=float, nargs='+', help='P0 values (percent or fraction) used only for testing')
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds = PulseData(a.data, seed=a.seed, holdout_p0=[v/100 if v > 1 else v for v in a.holdout_p0] if a.holdout_p0 else None)
    device = pick_device(a.device)
    print(f'{len(ds.raw["sample_id"])} usable samples: train {len(ds.train)}, val {len(ds.val)}, test {len(ds.test)}; '
          f'time input {ds.use_time}; device {device}')
    model = SpectrumToPulse(n_bins=ds.raw['spectrum'].shape[1], width=a.width, in_channels=ds.in_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    tr = [t.to(device) for t in ds.tensors(ds.train)]; va = [t.to(device) for t in ds.tensors(ds.val)]
    best, history, t0 = np.inf, [], time.time()
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(ds.train), device=device); tl = []
        for i in range(0, len(perm), a.batch):
            b = perm[i:i+a.batch]
            loss, _ = loss_fn(*model(tr[0][b]), tr[1][b], tr[2][b], tr[3][b])
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            ref = va if len(ds.val) else tr          # tiny datasets: fall back to the training set
            vl, vkl = loss_fn(*model(ref[0]), ref[1], ref[2], ref[3])
        history.append(dict(epoch=ep, train=float(np.mean(tl)), val=vl.item(), val_kl=vkl.item()))
        if vl.item() < best:
            best = vl.item(); torch.save(model.state_dict(), out/'model.pt')
        if ep % 25 == 0 or ep == a.epochs-1:
            print(f'epoch {ep:4d}  train {np.mean(tl):.4f}  val {vl.item():.4f} (KL {vkl.item():.4f})  {time.time()-t0:.0f}s', flush=True)
    model.load_state_dict(torch.load(out/'model.pt', map_location=device))

    # ---- physics evaluation on the test set
    if len(ds.test) == 0:
        print('No test samples; skipping the physics evaluation.')
        (out/'report.json').write_text(json.dumps(dict(data=str(a.data), args=vars(a), norm=ds.norm(),
                                                       best_val_loss=best, history=history), indent=1, default=float))
        return
    ev = PhysicsEvaluator(ds.manifest)
    test = ds.test[:a.eval_max]
    U_net, T_net = predict(model, ds, test, device)
    r = ds.raw
    mean_prog = (r['powers'][ds.train].mean(0), float(np.exp(np.log(r['duration'][ds.train]).mean())))
    Xtr = r['spectrum'][ds.train].astype(float)   # nearest neighbour on the spectrum (time is implicit in it)
    rows = []
    for k, i in enumerate(test):
        row = {f: r[f][i] for f in ('sample_id', 'material', 'p0', 'populations', 'powers', 'duration', 'state_Q')}
        nn = ds.train[int(np.argmin(((Xtr-r['spectrum'][i])**2).sum(1)))]
        scores = ev.score(row, {'network': (U_net[k], T_net[k]), 'mean_program': mean_prog,
                                'nearest_neighbour': (r['powers'][nn], float(r['duration'][nn]))})
        rows.append(scores)
    names = ['network', 'nearest_neighbour', 'mean_program']
    gain = lambda s, n: (s[n]-s['no_rf'])/max(s['optimizer']-s['no_rf'], 1e-9)
    summary = {'n_test_evaluated': len(rows),
               'Q_state': float(np.mean([s['state'] for s in rows])),
               'Q_no_rf': float(np.mean([s['no_rf'] for s in rows])),
               'Q_optimizer': float(np.mean([s['optimizer'] for s in rows]))}
    for n in names:
        g = np.array([gain(s, n) for s in rows])
        summary[n] = dict(Q=float(np.mean([s[n] for s in rows])), gain_fraction_mean=float(g.mean()),
                          gain_fraction_median=float(np.median(g)), gain_fraction_p10=float(np.percentile(g, 10)))
    report = dict(data=str(a.data), args=vars(a), norm=ds.norm(), splits=dict(train=len(ds.train), val=len(ds.val), test=len(ds.test)),
                  best_val_loss=best, history=history, physics=summary, per_sample=rows)
    (out/'report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
