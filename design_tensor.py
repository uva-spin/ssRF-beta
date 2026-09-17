"""Headless synchronized RF design or reproducible playback of an exported design."""
from pathlib import Path
import argparse
import json
import numpy as np
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot
from ssrf_realtime.tensor_optimizer import OptimizerSettings, optimize_tensor, replay, polarizations
from ssrf_realtime.pulse_program import PulseProgram


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--material',help='Full material JSON; without --state initializes its p0/q0')
    p.add_argument('--state',help='Current-state snapshot NPZ; preserves manipulated populations')
    p.add_argument('--out',default='tensor_design',help='New output directory (must not exist)')
    p.add_argument('--max-power',type=float,help='Override common per-bin U ceiling')
    p.add_argument('--duration-min',type=float)
    p.add_argument('--duration-max',type=float)
    p.add_argument('--budget',type=float,help='Search wall-time budget in seconds, excluding validation')
    p.add_argument('--replay',help='Replay an exported design directory; no optimization')
    a=p.parse_args()
    if a.replay:
        folder=Path(a.replay)
        s=PopulationSnapshot.load(folder/'starting_state.npz')
        prog=PulseProgram.load(folder/'rf_program.json')
        report=json.loads((folder/'optimization_report.json').read_text())
        n,_=replay(s,prog,report['common_duration'],dt=report['recommended_playback_dt'])
        with np.load(folder/'predicted_endpoint.npz',allow_pickle=False) as z:expected=z['n']
        check={'replayed':polarizations(n),'saved_prediction':report['final'],
               'max_population_difference':float(np.max(abs(n-expected)))}
        print(json.dumps(check,indent=2));return
    if Path(a.out).exists():p.error('Output directory already exists; choose a new --out path')
    config=MaterialConfig.load(a.material) if a.material else None
    if a.state:
        snapshot=PopulationSnapshot.load(a.state)
        if config and config.parameters!=snapshot.parameters:
            p.error('State and material parameters differ. Load/tune explicitly in the GUI and save a new snapshot.')
        model=snapshot.make_model()
    else:
        model=config.make_model(preserve_state=False) if config else IdealBinModel(IdealBinParams())
        snapshot=PopulationSnapshot.capture(model)
    settings=OptimizerSettings.from_dict(config.optimizer) if config else OptimizerSettings()
    for key,value in [('max_power',a.max_power),('min_duration',a.duration_min),('max_duration',a.duration_max),('max_wall_seconds',a.budget)]:
        if value is not None:setattr(settings,key,value)
    result=optimize_tensor(snapshot,settings,lambda text:print(text,flush=True))
    result.export(a.out,config)
    print(json.dumps(result.report,indent=2))
    print(f'Exported {Path(a.out).resolve()}')


if __name__=='__main__':main()
