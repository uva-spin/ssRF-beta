"""Plot an exported tensor design; does not run the optimizer or change settings."""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from ssrf_realtime.material_config import PopulationSnapshot


def plot_design(directory,show=False):
    directory=Path(directory)
    r=json.loads((directory/'optimization_report.json').read_text())
    s=PopulationSnapshot.load(directory/'starting_state.npz');m=s.make_model()
    with np.load(directory/'predicted_endpoint.npz',allow_pickle=False) as z:nf=z['n']
    data=np.genfromtxt(directory/'bin_diagnostics.csv',delimiter=',',names=True)
    R=data['R'];U=data['power_U']
    paths=[]
    fig=plt.figure(figsize=(9,4.2))
    plt.step(R,U,where='mid',label='Calculated per-bin RF command')
    plt.xlabel('physical R');plt.ylabel('RF command U [base-rate units]')
    plt.title(f"Synchronized RF profile — common duration T = {r['common_duration']:.5f}")
    plt.legend();plt.tight_layout()
    paths.append(directory/'synchronized_power_profile.png');fig.savefig(paths[-1],dpi=180)
    if show:plt.show()
    else:plt.close(fig)
    R,ip,im,total=m.spectrum_from_state(s.n)
    _,fp,fm,ft=m.spectrum_from_state(nf)
    fig=plt.figure(figsize=(9,4.5))
    plt.plot(R,ip,linestyle=':',label='Initial I+')
    plt.plot(R,im,linestyle=':',label='Initial I-')
    plt.plot(R,fp,label='Endpoint I+')
    plt.plot(R,fm,label='Endpoint I-')
    plt.xlabel('physical R');plt.ylabel('Intensity [display units]')
    plt.title(f"Full-model endpoint: P {r['initial']['P']:.3f} → {r['final']['P']:.3f}; "
              f"Q {r['initial']['Q']:.3f} → {r['final']['Q']:.3f}")
    plt.legend();plt.tight_layout()
    paths.append(directory/'predicted_endpoint_spectrum.png');fig.savefig(paths[-1],dpi=180)
    if show:plt.show()
    else:plt.close(fig)
    trace=np.genfromtxt(directory/'predicted_PQ_trace.csv',delimiter=',',names=True)
    fig=plt.figure(figsize=(9,4.2))
    plt.plot(trace['time_after_start'],trace['P'],label='Vector P')
    plt.plot(trace['time_after_start'],trace['Q'],label='Tensor Q')
    plt.xlabel('Time since program start [simulation units]');plt.ylabel('Polarization')
    plt.title('Playback through the unchanged population equations')
    plt.legend();plt.tight_layout()
    paths.append(directory/'predicted_PQ_evolution.png');fig.savefig(paths[-1],dpi=180)
    if show:plt.show()
    else:plt.close(fig)
    return paths

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',default=str(ROOT/'outputs'/'default_701_design'))
    args=parser.parse_args()
    for path in plot_design(args.directory):print(path)
