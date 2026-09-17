"""Generate an unoptimized horn + pedestal program and independent diagnostics.

Run: python examples/headless_demo.py
The script uses the same IdealBinModel as the GUI. It requires no Qt binding.
"""
from __future__ import annotations
from pathlib import Path
import sys
import csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.pulse_program import example_program


def main():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    params=IdealBinParams(rf_burn_R=-0.85)
    model=IdealBinModel(params)
    program=example_program(model.Rplus)
    program.save(ROOT/'examples'/'horn_and_pedestal.json')
    program.save_csv(ROOT/'examples'/'horn_and_pedestal.csv')
    model.set_program(program)
    initial=model.n.copy()
    model.start_program()
    rows=[]
    def record():
        obs=model.local_intensities(params.rf_burn_R)
        pol=model.polarizations()
        u=model.applied_rf_field()
        rows.append([model.t,model.program_elapsed,params.rf_burn_R,obs['Iplus'],obs['Iminus'],
                     pol['P'],pol['Q'],int(np.count_nonzero(u)),float(u.max())])
    record()
    while model.t < 3.5:
        model.step(6)
        record()
    columns=['simulation_time','program_time','monitor_R','Iplus','Iminus','P','Q','active_bins','max_base_rate']
    with (out/'ideal_profiles_time_trace.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(columns);writer.writerows(rows)
    planned=model.program.gain*model._compiled.exposure
    with (out/'ideal_profiles_exposure.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['bin_index','R','planned_base_exposure','delivered_base_exposure'])
        w.writerows(zip(range(len(model.Rplus)),model.Rplus,planned,model.delivered_exposure))
    error=float(np.max(np.abs(planned-model.delivered_exposure)))
    np.savez_compressed(out/'ideal_profiles_demo.npz',R=model.Rplus,initial=initial,final=model.n,
                        preview=model.preview_rf_field(),mu=model.mu,rows=np.array(rows),
                        planned_exposure=planned,delivered_exposure=model.delivered_exposure)
    # Each diagnostic is its own figure; no shared axes or subplot panels.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    c=program.compile()
    for time in (0.60,0.90,1.50,2.30):
        ax.step(model.Rplus,c.field_at(time),where='mid',label=f'program t={time:.2f}')
    ax.set(xlabel='Physical R',ylabel='Commanded RF base rate U',title='Two independently timed ideal RF profiles')
    ax.legend();fig.tight_layout();fig.savefig(out/'ideal_profiles_commands.png',dpi=160);plt.close(fig)
    trace=np.array(rows)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(trace[:,0],trace[:,3],label='I+(R,t)')
    ax.plot(trace[:,0],trace[:,4],label='I-(R,t)')
    ax.set(xlabel='Simulation time',ylabel='Absolute intensity [arb.]',title=f'Local intensities at monitor R={params.rf_burn_R:g}')
    ax.legend();fig.tight_layout();fig.savefig(out/'ideal_profiles_local_intensities.png',dpi=160);plt.close(fig)
    print(f'Created example JSON/CSV and diagnostics in {out}')
    print(f'Maximum absolute exposure error: {error:.3e}')
    print(f'Program finished: {model.program_state}; simulation t={model.t:.4f}')
    assert error<1e-11


if __name__=='__main__':main()
