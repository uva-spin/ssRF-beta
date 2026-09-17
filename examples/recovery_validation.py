"""Reproduce the old mirror failure and validate revised finite-rate recovery.

Requires SciPy (requirements-dev.txt). This integrates exactly the model's
population derivative with an independent adaptive solver, not a target-reset
or a prescribed spectral recovery curve. The RF preparation uses the real
per-bin scheduler. All tests operate on synthetic data, not measurements.

Usage: python examples/recovery_validation.py [--bins 701] [--end 2500]
"""
from __future__ import annotations
from pathlib import Path
import argparse, csv, json, sys, time
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.pulse_program import BinPulse, RFProfile, PulseProgram


def run_case(initial, params, times):
    m=IdealBinModel(params)
    m.n=initial.copy();m.stop_program();m.set_diffusion_enabled(True)
    shape=m.n.shape
    def f(t,y):
        m.n=y.reshape(shape)
        return m.derivative(rf_on=False,dnp_on=False).ravel()
    tic=time.perf_counter()
    sol=solve_ivp(f,(0,float(times[-1])),m.n.ravel(),t_eval=times,
                  method='DOP853',rtol=2e-10,atol=8e-15)
    if not sol.success:raise RuntimeError(sol.message)
    states=sol.y.T.reshape((-1,)+shape)
    m.n=states[-1].copy()
    P=(states[:,:,0]-states[:,:,2]).sum(axis=1)
    Q=(states[:,:,0]-2*states[:,:,1]+states[:,:,2]).sum(axis=1)
    target=m.equilibrium_reference(float(P[0]))
    fractional_error=(states-target[None,:,:])/m.mu[None,:,None]
    rms=np.sqrt(np.sum(m.mu[None,:,None]*fractional_error**2,axis=(1,2)))
    frac=states/m.mu[None,:,None]
    safe=np.maximum(frac,np.finfo(float).tiny)
    entropy=-np.sum(states*np.log(safe),axis=(1,2))
    return m,states,P,Q,rms,entropy,dict(nfev=sol.nfev,seconds=time.perf_counter()-tic,
        max_P_drift=float(np.max(abs(P-P[0]))),
        max_packet_mass_error=float(np.max(abs(states.sum(axis=2)-m.mu))),
        final_Q=float(Q[-1]),final_Q_B=float(m.polarizations()['Q_boltz_at_P']),
        final_Q_error=float(Q[-1]-m.polarizations()['Q_boltz_at_P']),
        final_population_rms=float(rms[-1]),
        final_max_fraction_error=float(np.max(abs(fractional_error[-1]))),
        minimum_population=float(states.min()),
        min_entropy_increment=float(np.min(np.diff(entropy))))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bins',type=int,default=701)
    ap.add_argument('--end',type=float,default=2500.)
    args=ap.parse_args()
    out=ROOT/'outputs'/'recovery_validation';out.mkdir(parents=True,exist_ok=True)
    m=IdealBinModel(IdealBinParams(n_bins=args.bins,diffusion_enabled=False,dt=.0015))
    R_requested=.4;k=m.branch_indices(R_requested)[0];R=float(m.Rplus[k])
    m.params.rf_burn_R=R
    initial=m.n.copy();P_initial=m.polarizations()['P']
    prog=PulseProgram(len(m.Rplus),m.Rplus[0],m.Rplus[-1],
                     profiles=[RFProfile('controlled one-bin burn',pulses=[BinPulse(k,.3,0.,1.)])])
    m.set_program(prog);m.start_program();m.step(667)
    assert m.program_state=='finished'
    burned=m.n.copy();P_after=m.polarizations()['P']
    assert P_after<P_initial
    times=np.unique(np.r_[0,np.geomspace(.01,80,160),np.geomspace(80,args.end,120)])
    new=run_case(burned,IdealBinParams(n_bins=args.bins,rf_burn_R=R),times)
    oldtimes=times[times<=80]
    old=run_case(burned,IdealBinParams(n_bins=args.bins,rf_burn_R=R,
                 cross_branch_ratio=0,double_quantum_ratio=0,
                 diffusion_overlap='gaussian',kernel_cutoff_widths=4),oldtimes)
    mnew,states,P,Q,rms,entropy,stats=new
    oldstates=old[1]
    kp,km=mnew.branch_indices(R);mu=mnew.mu
    def local_features(a):
        p=a/mu[None,:,None];Ip=p[:,:,0]-p[:,:,1];Im=p[:,:,1]-p[:,:,2]
        return np.stack([Ip[:,km]-.5*(Ip[:,km-1]+Ip[:,km+1]),
                         Im[:,kp]-.5*(Im[:,kp-1]+Im[:,kp+1])],axis=1)
    cnew=local_features(states);cold=local_features(oldstates)
    factor=mnew.display_cal/mnew.dR
    visible_mirrors=np.stack([(states[:,km,0]-states[:,km,1])*factor,
                             (states[:,kp,1]-states[:,kp,2])*factor],axis=1)
    target=mnew.equilibrium_reference(P_after)
    summary={'description':'Controlled synthetic audit, no DNP/T1 during recovery; RF history exactly shared.',
             'n_bins':args.bins,'requested_R':R_requested,'actual_R':R,'P_initial':P_initial,
             'P_after_RF':P_after,'RF_area_loss':mnew.display_cal*(P_initial-P_after),
             'RF_per_bin_exposure':float(m.delivered_exposure[k]),
             'end_recovery_time':args.end,'new':stats,'old_at_80':old[-1],
             'old_min_local_contrast_fraction':np.min(cold/cold[0],axis=0).tolist(),
             'new_min_local_contrast_fraction_first_80':np.min(cnew[:len(oldtimes)]/cnew[0],axis=0).tolist(),
             'parameters':vars(mnew.params),
             'note':'Equilibrium endpoint and numerical consistency are tested; rate calibration is not established.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/'recovery_comparison.npz',times=times,oldtimes=oldtimes,
                        old_contrast=cold,new_contrast=cnew,P=P,Q=Q,rms=rms,entropy=entropy,
                        initial=initial,burned=burned,final=states[-1],target=target,
                        R=mnew.Rplus,mu=mu,display_cal=mnew.display_cal,dR=mnew.dR,
                        mirrors=visible_mirrors,reference_P=P_after,
                        reference_Q=mnew.polarizations(target)['Q'])
    with (out/'recovery_trace.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time','P','Q','Q_B_at_remaining_P','population_RMS','entropy',
                                  'Iplus_mirror','Iminus_mirror','local_contrast_plus','local_contrast_minus'])
        for idx,t in enumerate(times):
            w.writerow([t,P[idx],Q[idx],mnew.polarizations(target)['Q'],rms[idx],entropy[idx],
                        *visible_mirrors[idx],*cnew[idx]])
    assert stats['max_P_drift']<1e-9
    assert stats['max_packet_mass_error']<1e-10
    assert stats['minimum_population']>0
    assert stats['final_population_rms']<1e-7
    assert abs(stats['final_Q_error'])<1e-7
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
