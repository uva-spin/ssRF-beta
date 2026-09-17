"""Synchronized ideal-bin tensor-profile design from a frozen population state.

All active bins receive a constant command on [0,T), with one shared T. The
objective is positive Q immediately at T, not |Q|, and includes all currently
enabled material mechanisms. This is a bounded local search, NOT a certificate
of a global optimum. The core simulator is never mutated.

Forward kinetics below are an algebraic regrouping of model.py, with an analytic
discrete-adjoint gradient. Each delivered result is replayed through the unchanged
IdealBinModel, and checked again with half the integration step. No mirror rule,
Boltzmann reset, signal convolution, or altered recovery operator is introduced.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields, field
from pathlib import Path
from typing import Callable, Optional
import csv
import math
import time
import json
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.optimize import minimize

from .pulse_program import BinPulse, RFProfile, PulseProgram
from .material_config import PopulationSnapshot, MaterialConfig, write_json


class OptimizationCancelled(RuntimeError): pass


@dataclass
class OptimizerSettings:
    max_power: float = 20.0          # Trial command ceiling; NOT watts or a material constant.
    min_duration: float = 0.01
    max_duration: float = 2.0
    duration_samples: int = 7
    duration_refinements: int = 3
    max_iterations: int = 30
    starts: int = 2
    search_dt: float = 0.004
    max_steps: int = 20000
    max_validation_halvings: int = 6
    max_wall_seconds: float = 300.0
    q_absolute_tolerance: float = 1e-6
    q_relative_tolerance: float = 0.002
    convergence_tolerance: float = 2e-5
    candidate_mode: str = 'union'   # Negative signed subtraction OR positive initial RF slope.
    candidate_threshold: float = 1e-10
    regions: list = field(default_factory=list)       # Empty = entire spectrum; otherwise [[Rlo,Rhi],...].
    per_bin_power_limits: dict = field(default_factory=dict)  # Optional 0-based bin string -> U ceiling.
    seed: int = 42

    def validate(self):
        for f in fields(self):
            value=getattr(self,f.name)
            if f.name in ('regions','per_bin_power_limits','candidate_mode'): continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'Optimizer {f.name} must be a finite number')
        for key in ('duration_samples','duration_refinements','max_iterations','starts','max_steps','max_validation_halvings','seed'):
            v=getattr(self,key)
            if not isinstance(v,int): raise ValueError(f'{key} must be an integer')
        if self.max_power<=0: raise ValueError('max_power must be positive')
        if not 0<self.min_duration<=self.max_duration: raise ValueError('Require 0 < min_duration <= max_duration')
        if not 2<=self.duration_samples<=40: raise ValueError('duration_samples must be 2..40')
        if not 0<=self.duration_refinements<=20: raise ValueError('duration_refinements must be 0..20')
        if not 1<=self.max_iterations<=1000 or not 1<=self.starts<=10: raise ValueError('Invalid iterations/starts')
        if not 2<=self.max_steps<=100000: raise ValueError('max_steps must be 2..100000')
        if not 0<=self.max_validation_halvings<=12: raise ValueError('max_validation_halvings must be 0..12')
        if self.seed<0: raise ValueError('seed must be nonnegative')
        for key in ('search_dt','max_wall_seconds','convergence_tolerance'):
            if getattr(self,key)<=0: raise ValueError(f'{key} must be positive')
        if self.q_absolute_tolerance<0 or not 0<=self.q_relative_tolerance<1 or self.candidate_threshold<0:
            raise ValueError('Invalid Q tolerance or candidate threshold')
        if self.candidate_mode not in ('union','tensor_deficit','rf_gain','all'):
            raise ValueError('Unknown candidate mode')
        if not isinstance(self.regions,list): raise ValueError('regions must be a list')
        for pair in self.regions:
            if not isinstance(pair,(tuple,list)) or len(pair)!=2 or not all(
                isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in pair) or pair[0]>pair[1]:
                raise ValueError('Each region must be [R_left, R_right]')
        if not isinstance(self.per_bin_power_limits,dict): raise ValueError('per_bin_power_limits must be an object')
        for k,v in self.per_bin_power_limits.items():
            if not isinstance(k,str) or not k.isdigit() or str(int(k))!=k:
                raise ValueError('Per-bin limits require canonical 0-based integer string keys')
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=self.max_power:
                raise ValueError('Per-bin power ceilings must be finite and in [0,max_power]')
        return self

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or set(data)!={f.name for f in fields(cls)}:
            raise ValueError('Optimizer settings must contain exactly the schema fields')
        return cls(**data).validate()


def polarizations(n):
    return {'P':float(np.sum(n[:,0]-n[:,2])),
            'Q':float(np.sum(n[:,0]-2*n[:,1]+n[:,2]))}


class FrozenDynamics:
    """Unchanged population ODE, regrouped for fast many-trial evaluations.

    For uncorrelated EFG/spatial geometry the SQ kernels are Toeplitz/Hankel;
    FFT products are mathematically the same pair sums (all finite-grid tails,
    exact optional cutoff, and half self-packet cross weight retained).
    For orientation correlation a direct matrix representation is used.
    """
    def __init__(self,snapshot: PopulationSnapshot):
        self.snapshot=snapshot
        self.model=snapshot.make_model()
        self.p=self.model.params
        self.mu=snapshot.mu.copy(); self.n0=snapshot.n.copy(); self.N=len(self.mu)
        self.w=self.model.capacity_rate_weights().copy()
        self.scale=(self.p.diffusion_scale*(self.p.microwave_diffusion_factor if self.p.dnp_enabled else 1.0)
                    if self.p.diffusion_enabled else 0.0)
        self.kx=self.scale*self.p.cross_branch_ratio
        self.kdq=self.scale*self.p.double_quantum_ratio
        self.decay=np.full(self.N,self.p.t1_rate)
        self.source=self.p.t1_rate*self.model.equilibrium_reference(self.p.t1_p_eq)
        if self.p.dnp_enabled:
            d=self.p.dnp_rate*self.w
            self.decay+=d
            self.source+=d[:,None]*self.model.equilibrium_reference(self.p.p_dnp_sat)
        self._fft=None; self.H=self.X=self.C=None
        if self.scale:
            R=snapshot.grid
            # At an exactly-on-grid hard cutoff the original floating-point
            # distance comparisons need not be Toeplitz. Use the exact matrix.
            cut_bins=self.p.kernel_cutoff_widths*self.p.zq_width_R/self.model.dR
            cutoff_ambiguous=(self.p.kernel_cutoff_widths>0 and abs(cut_bins-round(cut_bins))<1e-9)
            if self.p.orientation_corr_fraction == 0 and not cutoff_ambiguous:
                col=self.model._spectral_overlap(np.arange(self.N)*self.model.dR)
                self.L=next_fast_len(2*self.N-1)
                embed=np.zeros(self.L); embed[:self.N]=col; embed[-self.N+1:]=col[1:][::-1]
                self._fft=rfft(embed)[:,None]
                self.cross_diag=.5*self.model._spectral_overlap(2*R)
            else:
                theta=self.model._effective_theta()
                f=self.p.orientation_corr_fraction
                sig=np.deg2rad(self.p.orientation_corr_width_deg)
                C=(1-f)+f*np.exp(-.5*((theta[:,None]-theta[None,:])/sig)**2)
                self.H=C*self.model._spectral_overlap(R[:,None]-R[None,:])
                self.X=C*self.model._spectral_overlap(R[:,None]+R[None,:])
                self.X[np.diag_indices(self.N)]*=.5
                self.C=C.copy(); np.fill_diagonal(self.C,0)

    def h(self,y):
        if self._fft is not None:
            return irfft(self._fft*rfft(y,n=self.L,axis=0),n=self.L,axis=0)[:self.N]
        return self.H@y

    def x(self,y):
        if self._fft is not None: return self.h(y[::-1])-self.cross_diag[:,None]*y
        return self.X@y

    def c(self,y):
        return np.sum(y,axis=0)[None,:] if self.C is None else self.C@y

    def rhs(self,n,u):
        gp=self.w*u; gm=self.w*u[::-1]
        jp=gp*(n[:,0]-n[:,1]); jm=gm*(n[:,1]-n[:,2])
        out=-self.decay[:,None]*n+self.source
        out[:,0]-=jp; out[:,1]+=jp-jm; out[:,2]+=jm
        if self.scale:
            Hn=self.h(n)
            a=self.scale*(n[:,1]*Hn[:,0]-n[:,0]*Hn[:,1])
            b=self.scale*(n[:,2]*Hn[:,1]-n[:,1]*Hn[:,2])
            out[:,0]+=a; out[:,1]+=-a+b; out[:,2]-=b
            if self.kx:
                Xn=self.x(n)
                c=self.kx*(n[:,1]*Xn[:,1]-n[:,0]*Xn[:,2])
                d=self.kx*(n[:,1]*Xn[:,1]-n[:,2]*Xn[:,0])
                out[:,0]+=c; out[:,1]-=c+d; out[:,2]+=d
            if self.kdq:
                Cn=self.c(n)
                z=self.kdq*(n[:,2]*Cn[:,0]-n[:,0]*Cn[:,2])
                out[:,0]+=z; out[:,2]-=z
        return out

    def vjp(self,n,u,adj):
        """Exact state-Jacobian transpose and command gradient of adj dot rhs."""
        va=adj[:,0]-adj[:,1]; vb=adj[:,1]-adj[:,2]
        vd=-vb; vz=adj[:,0]-adj[:,2]
        gp=self.w*u; gm=self.w*u[::-1]
        g=-self.decay[:,None]*adj
        g[:,0]-=gp*va; g[:,1]+=gp*va-gm*vb; g[:,2]+=gm*vb
        gu=-self.w*(n[:,0]-n[:,1])*va-(self.w*(n[:,1]-n[:,2])*vb)[::-1]
        if self.scale:
            s=self.scale; Hn=self.h(n)
            g[:,0]-=s*va*Hn[:,1]
            g[:,1]+=s*(va*Hn[:,0]-vb*Hn[:,2])
            g[:,2]+=s*vb*Hn[:,1]
            b=np.zeros_like(n)
            b[:,0]=s*va*n[:,1]; b[:,1]=s*(-va*n[:,0]+vb*n[:,2]); b[:,2]=-s*vb*n[:,1]
            g+=self.h(b)
            if self.kx:
                s=self.kx; Xn=self.x(n)
                g[:,0]-=s*va*Xn[:,2]; g[:,1]+=s*(va+vd)*Xn[:,1]; g[:,2]-=s*vd*Xn[:,0]
                b[:,0]=-s*vd*n[:,2]; b[:,1]=s*(va+vd)*n[:,1]; b[:,2]=-s*va*n[:,0]
                g+=self.x(b)
            if self.kdq:
                s=self.kdq; Cn=self.c(n)
                g[:,0]-=s*vz*Cn[:,2]; g[:,2]+=s*vz*Cn[:,0]
                b.fill(0); b[:,0]=s*vz*n[:,2]; b[:,2]=-s*vz*n[:,0]
                g+=self.c(b)
        return g,gu

    def outgoing_bound(self,upper):
        out=self.w*(upper+upper[::-1])+self.decay
        if self.scale:
            mu=self.mu[:,None]
            # Same-packet self exchanges cancel; retain them as a safe overbound.
            out+=2*self.scale*self.h(mu)[:,0]
            if self.kx: out+=2*self.kx*self.x(mu)[:,0]
            if self.kdq: out+=self.kdq*np.broadcast_to(self.c(mu),(self.N,1))[:,0]
        return float(out.max())

    def evaluate(self,u,T,dt,max_steps=20000,gradient=False,check=None):
        """Fixed-step explicit Euler; never clips a significant negative population."""
        steps=max(1,int(np.ceil(T/dt)))
        if steps>max_steps: raise ValueError(f'{steps} prediction steps exceed max_steps={max_steps}; '
                                            'reduce duration/power or increase the limit')
        if gradient and steps*self.N*3*8 > 512*1024**2:
            raise ValueError('Adjoint trajectory exceeds 512 MiB; reduce duration/grid or use a smaller search.')
        h=T/steps
        n=self.n0.copy(); history=[] if gradient else None
        for it in range(steps):
            if check and it%32==0: check()
            if gradient: history.append(n)
            n=n+h*self.rhs(n,u)
        if not np.isfinite(n).all() or np.any(n < -1e-11*self.mu[:,None]):
            raise FloatingPointError('Prediction left the population simplex; reduce search_dt')
        Q=polarizations(n)['Q']
        if not gradient: return Q,n
        adj=np.broadcast_to(np.array([1.,-2.,1.]),n.shape).copy(); gu=np.zeros(self.N)
        for it in range(steps-1,-1,-1):
            if check and it%32==0: check()
            gn,gc=self.vjp(history[it],u,adj)
            gu+=h*gc; adj+=h*gn
        return Q,n,gu


def candidate_arrays(dyn,settings):
    R=dyn.snapshot.grid; n=dyn.n0; w=dyn.w
    dp=n[:,0]-n[:,1]; dm=(n[:,1]-n[:,2])[::-1]
    signed_tensor=(dp-dm)/dyn.model.dR
    gain=3*(w[::-1]*dm-w*dp)
    t=settings.candidate_threshold
    deficit=signed_tensor < -t*max(float(np.max(abs(signed_tensor))),1e-30)
    favorable=gain>t*max(float(np.max(abs(gain))),1e-30)
    mask={'tensor_deficit':deficit,'rf_gain':favorable,'union':deficit|favorable,
          'all':np.ones(len(R),dtype=bool)}[settings.candidate_mode].copy()
    if settings.regions:
        region=np.zeros(len(R),dtype=bool)
        for lo,hi in settings.regions: region|=(R>=lo)&(R<=hi)
        mask&=region
    upper=np.full(len(R),settings.max_power)
    for key,value in settings.per_bin_power_limits.items():
        j=int(key)
        if j>=len(R): raise ValueError(f'Per-bin ceiling {j} is outside the grid')
        upper[j]=value
    upper[~mask]=0
    mask&=upper>0
    return mask,upper,signed_tensor,gain


def rf_only_seed(dyn,T,upper):
    """Isolated noncentral-bin exposure maxima give only a starting guess.

    Candidate endpoints and a positive stationary exposure are compared. For
    simultaneous mirror commands, the full ODE optimization, not this seed,
    determines the answer. Signs are never inferred merely from global P.
    """
    n=dyn.n0; a=n[:,0]-n[:,1]; b=(n[:,1]-n[:,2])[::-1]
    cp=dyn.w; cm=dyn.w[::-1]; Emax=upper*T
    E=np.stack((np.zeros_like(Emax),Emax,.5*Emax))
    with np.errstate(divide='ignore',invalid='ignore'):
        stationary=np.log((cm*b)/(cp*a))/(2*(cm-cp))
    valid=np.isfinite(stationary)&(stationary>0)&(stationary<Emax)&(cp*a>0)&(cm*b>0)
    E=np.vstack((E,np.where(valid,stationary,0)))
    gains=1.5*(b[None,:]*(-np.expm1(-2*cm[None,:]*E))-
                a[None,:]*(-np.expm1(-2*cp[None,:]*E)))
    pick=np.argmax(gains,axis=0)
    return E[pick,np.arange(dyn.N)]/T


def replay(snapshot,program,T,dt=None,check=None):
    """Independent prediction through the unchanged scheduler and model."""
    m=snapshot.make_model()
    m.set_program(program)
    if dt is not None: m.params.dt=dt
    initial_dt=m.params.dt
    if m._compiled.end_time>0: m.start_program(True)
    else: m.stop_program()
    t0=m.t; trace=[]
    P=polarizations(m.n); trace.append((0.,P['P'],P['Q']))
    count=0
    while m.t-t0 < T:
        if check and count%32==0: check()
        remaining=T-(m.t-t0)
        if remaining<1e-14*max(1,T): break
        m.params.dt=min(initial_dt,remaining)
        m.step(1)
        count+=1
        if count%max(1,int(T/initial_dt/300))==0:
            P=polarizations(m.n); trace.append((m.t-t0,P['P'],P['Q']))
    P=polarizations(m.n)
    trace.append((T,P['P'],P['Q']))
    m.params.dt=initial_dt
    return m.n.copy(),np.asarray(trace)


@dataclass
class DesignResult:
    snapshot: PopulationSnapshot
    settings: OptimizerSettings
    program: PulseProgram
    duration: float
    powers: np.ndarray
    final_n: np.ndarray
    no_rf_n: np.ndarray
    trace: np.ndarray
    candidates: np.ndarray
    signed_tensor: np.ndarray
    initial_gain: np.ndarray
    report: dict
    scan: list

    def export(self,directory,material=None):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        self.program.save(directory/'rf_program.json')
        self.program.save_csv(directory/'rf_program.csv')
        self.snapshot.save(directory/'starting_state.npz')
        self.snapshot.make_model()  # Validate that it is independently replayable.
        material=material or MaterialConfig.capture(self.snapshot.make_model(),optimizer=self.settings)
        # Always record parameters of the DESIGN SNAPSHOT, not a subsequently edited live material.
        material=MaterialConfig(material.name,dict(self.snapshot.parameters),asdict(self.settings),
                                dict(material.gui),material.description)
        material.save(directory/'material.json')
        write_json(directory/'optimization_report.json',self.report)
        with (directory/'duration_search.csv').open('w',newline='',encoding='utf-8') as f:
            columns=['duration','Q_search','Q_no_RF','iterations','evaluations','converged','message']
            wr=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore'); wr.writeheader(); wr.writerows(self.scan)
        with (directory/'predicted_PQ_trace.csv').open('w',newline='',encoding='utf-8') as f:
            wr=csv.writer(f); wr.writerow(['time_after_start','P','Q']); wr.writerows(self.trace)
        model=self.snapshot.make_model()
        R,ap,am,_=model.spectrum_from_state(self.snapshot.n)
        _,bp,bm,_=model.spectrum_from_state(self.final_n)
        with (directory/'bin_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
            wr=csv.writer(f); wr.writerow(['bin_index','R','candidate','power_U','start','duration','exposure',
                'initial_signed_tensor_density','initial_RF_dQdt_per_U','Iplus_initial','Iminus_initial',
                'Iplus_final','Iminus_final'])
            for j in range(len(R)):
                wr.writerow([j,R[j],int(self.candidates[j]),self.powers[j],0.,self.duration,
                    self.powers[j]*self.duration,self.signed_tensor[j],self.initial_gain[j],ap[j],am[j],bp[j],bm[j]])
        with (directory/'predicted_endpoint.npz').open('wb') as f:
            np.savez_compressed(f,n=self.final_n,no_rf_n=self.no_rf_n,grid=R,mu=model.mu)
        (directory/'README.txt').write_text(
            'Synchronized ideal-bin tensor design\n\n'
            'Load rf_program.json (or .csv) in the existing pulse editor. All nonzero rows start at 0 '
            'and end at one common duration. U is an effective base rate, NOT calibrated watts.\n'
            'Load starting_state.npz to reproduce the input population state and material kinetics.\n'
            'Generating/loading the program does not run it. Press Start / restart program explicitly.\n'
            'A different live state or material does not have the same prediction.\n'
            'optimization_report.json contains search limits, convergence checks and no-RF comparison.\n'
            'This is a finite bounded local search; no global optimum is certified.\n',encoding='utf-8')
        return directory


def synchronized_program(grid,powers,T):
    p=PulseProgram(n_bins=len(grid),r_min=float(grid[0]),r_max=float(grid[-1]),gain=1.)
    for title,mask in [('Negative-R region',grid<0),('Nonnegative-R region',grid>=0)]:
        rows=[BinPulse(int(j),float(powers[j]),0.,float(T)) for j in np.flatnonzero(mask&(powers>0))]
        if rows: p.profiles.append(RFProfile(title,pulses=rows))
    if not p.profiles:
        p.profiles=[RFProfile('No beneficial RF selected',pulses=[BinPulse(0,0.,0.,0.,False)])]
    p.validate()
    return p


def optimize_tensor(snapshot,settings=None,progress=None,cancel=None):
    """Bounded per-bin L-BFGS-B powers, duration scan and shortest-near-best search.

    All material mechanisms follow the captured flags. A zero-power control is
    always considered; no-RF endpoints are reported to avoid crediting DNP or
    natural recovery to RF. Candidate mask is explicit and saved.
    """
    settings=(settings or OptimizerSettings()).validate()
    start_clock=time.monotonic(); deadline=start_clock+settings.max_wall_seconds
    budget_hit=False
    def check():
        if cancel is not None and cancel.is_set(): raise OptimizationCancelled('Design cancelled; live state unchanged')
    def search_check():
        check()
        if time.monotonic()>deadline: raise TimeoutError('Search time budget reached')
    def notify(text):
        if progress: progress(text)
    dyn=FrozenDynamics(snapshot)
    mask,upper,qsign,gain=candidate_arrays(dyn,settings)
    ix=np.flatnonzero(mask)
    bound=dyn.outgoing_bound(upper)
    dt=min(settings.search_dt,.18/bound if bound>0 else settings.search_dt)
    if math.ceil(settings.max_duration/dt)>settings.max_steps:
        raise ValueError('Requested power/duration needs more prediction steps than max_steps permits')
    scan=[]; choices=[]; warm=None; rng=np.random.default_rng(settings.seed)
    initial=polarizations(snapshot.n)
    notify(f'Captured P={initial["P"]:+.6f}, Q={initial["Q"]:+.6f}; {len(ix)} candidate bins. '
           f'Material dynamics retained. Search step <= {dt:.5g}.')
    def at_duration(T):
        nonlocal warm
        search_check()
        zero=np.zeros(dyn.N)
        Qzero,_=dyn.evaluate(zero,T,dt,settings.max_steps,check=search_check)
        bestU=zero; bestQ=Qzero
        neval=0; nit=0; converged=True; messages=[]
        starts=[rf_only_seed(dyn,T,upper)]
        if warm is not None: starts.insert(0,warm.copy())
        while len(starts)<settings.starts:
            starts.append(upper*(.25+.75*rng.random(dyn.N)))
        # First seed is analytic on the first horizon and continuation thereafter;
        # include an independent seed on every horizon when starts>=2.
        if len(ix):
            for seed in starts[:settings.starts]:
                def objective(z):
                    nonlocal neval,bestQ,bestU
                    u=np.zeros(dyn.N); u[ix]=upper[ix]*z
                    Q,_,grad=dyn.evaluate(u,T,dt,settings.max_steps,True,search_check)
                    neval+=1
                    if Q>bestQ: bestQ=Q; bestU=u.copy()
                    # Scale Q so small per-bin gradients are not mistaken for convergence.
                    return -1000*Q,-1000*grad[ix]*upper[ix]
                z=np.clip(seed[ix]/upper[ix],0,1)
                result=minimize(objective,z,method='L-BFGS-B',jac=True,bounds=[(0.,1.)]*len(ix),
                    options={'maxiter':settings.max_iterations,'gtol':1e-6,'ftol':1e-11,'maxls':15})
                nit+=int(result.nit); converged=converged and bool(result.success)
                messages.append(str(result.message))
        else: messages=['No candidate bins; no-RF endpoint only']
        warm=bestU.copy()
        item={'duration':float(T),'Q_search':float(bestQ),'Q_no_RF':float(Qzero),'iterations':nit,
              'evaluations':neval,'converged':converged,'message':'; '.join(messages)}
        scan.append(item); choices.append((float(T),bestU.copy(),float(bestQ)))
        notify(f'T={T:.6g}: Q={bestQ:+.7f}; no RF {Qzero:+.7f}; {neval} gradient evaluations.')
        return choices[-1]
    durations=np.unique(np.geomspace(settings.min_duration,settings.max_duration,settings.duration_samples))
    try:
        for T in durations: at_duration(float(T))
        # Refine between earliest near-best horizon and its shorter neighbor;
        # this is a local refinement, not a monotonicity or global certificate.
        for _ in range(settings.duration_refinements):
            best=max(c[2] for c in choices)
            tol=max(settings.q_absolute_tolerance,settings.q_relative_tolerance*max(0,best-initial['Q']))
            eligible=sorted([c for c in choices if c[2]>=best-tol],key=lambda c:c[0])
            first=eligible[0]; below=sorted([c[0] for c in choices if c[0]<first[0]])
            if not below: break
            at_duration(math.sqrt(below[-1]*first[0]))
    except TimeoutError:
        budget_hit=True; notify('Search budget reached; validating completed horizon candidates.')
    if not choices:
        raise RuntimeError('No horizon completed before the time budget. Reduce samples/iterations, shorten '
                           'the time range, or increase the budget.')
    check()
    # Refine numerical endpoint accuracy for EVERY completed candidate before
    # choosing shortest near-best. Never select by a coarse Q alone.
    fine=[]
    dt_fine=min(snapshot.parameters['dt'],dt)
    for T,u,Q in choices:
        Qf,nf=dyn.evaluate(u,T,dt_fine,settings.max_steps,check=check)
        fine.append((T,u,Qf,nf))
    best=max(c[2] for c in fine)
    tol=max(settings.q_absolute_tolerance,settings.q_relative_tolerance*max(0,best-initial['Q']))
    # Do nothing now is also an option. Do not generate damaging irradiation.
    if best<=initial['Q']+settings.q_absolute_tolerance:
        T=0.; u=np.zeros(dyn.N); endpoint=snapshot.n.copy(); control=snapshot.n.copy()
        trace=np.array([[0.,initial['P'],initial['Q']]])
        errQ=0.; errState=0.; replay_diff=0.; best_fine=float(best)
        playback_dt=float(snapshot.parameters['dt']); validation_halvings=0
        status='no_improvement'
    else:
        selected=min((c for c in fine if c[2]>=best-tol),key=lambda c:c[0])
        T,u,Q,nf=selected
        program=synchronized_program(snapshot.grid,u,T)
        notify(f'Validating T={T:.6g} through the unchanged program scheduler and material model...')
        playback_dt=float(snapshot.parameters['dt'])
        validation_halvings=0
        # Keep refining numerical resolution, never the material kinetics or RF
        # powers, until the exported endpoint prediction is time-step stable.
        while True:
            endpoint,trace=replay(snapshot,program,T,dt=playback_dt,check=check)
            half_dt=min(dt_fine,playback_dt/2)
            _,half=dyn.evaluate(u,T,half_dt,
                                settings.max_steps*(2**(settings.max_validation_halvings+1)),check=check)
            errQ=abs(polarizations(endpoint)['Q']-polarizations(half)['Q'])
            errState=float(np.sqrt(np.sum(snapshot.mu[:,None]*((endpoint-half)/snapshot.mu[:,None])**2)))
            if (errQ<=settings.convergence_tolerance and errState<=10*settings.convergence_tolerance
                    or validation_halvings>=settings.max_validation_halvings): break
            playback_dt*=.5; validation_halvings+=1
            notify(f'Refining playback dt to {playback_dt:.6g}; previous Q check {errQ:.3g}.')
        _,control=dyn.evaluate(np.zeros(dyn.N),T,min(dt_fine,playback_dt),
                               settings.max_steps*(2**settings.max_validation_halvings),check=check)
        replay_diff=abs(polarizations(endpoint)['Q']-Q)
        best_fine=float(best)
        status='completed'
        if errQ>settings.convergence_tolerance or errState>10*settings.convergence_tolerance:
            status='needs_finer_time_step'
        if polarizations(endpoint)['Q']<initial['Q']-settings.convergence_tolerance:
            status='needs_finer_time_step'
        if not np.any(u): status='no_RF_benefit'
    program=synchronized_program(snapshot.grid,u,T)
    final=polarizations(endpoint); nr=polarizations(control)
    warnings=[]
    if validation_halvings: warnings.append('Playback requires the smaller validated dt recorded in this report. The GUI applies it only when loading this design.')
    if budget_hit: warnings.append('Time budget reached before completing the requested search.')
    if any(not v['converged'] for v in scan): warnings.append('Some local power optimizations reached a limit or did not converge.')
    if status=='needs_finer_time_step': warnings.append('Endpoint time-step validation failed; reduce dt/search_dt and regenerate before applying.')
    if T and abs(T-settings.max_duration)<1e-12: warnings.append('Chosen duration is at the upper search bound; consider extending the range.')
    if snapshot.parameters['dnp_enabled'] or snapshot.parameters['t1_rate'] or snapshot.parameters['diffusion_enabled']:
        warnings.append('Objective is final Q, not Q minus the no-RF endpoint. Natural dynamics may contribute to the increase.')
    if not np.any(u): warnings.append('No nonzero RF command selected; do not interpret natural buildup as RF enhancement.')
    report={'schema':'ssrf.tensor_design.v1','status':status,'objective':'maximize positive Q at common endpoint; shortest within tolerance',
        'search_scope':'finite horizon scan + bounded local L-BFGS-B power optimization; no global certificate',
        'snapshot_fingerprint':snapshot.fingerprint,'snapshot_simulation_time':snapshot.simulation_time,
        'initial':initial,'final':final,'no_RF_at_same_endpoint':nr,
        'delta_Q':final['Q']-initial['Q'],'RF_advantage_over_no_RF':final['Q']-nr['Q'],
        'delta_P':final['P']-initial['P'],'common_start':0.,'common_duration':float(T),
        'active_bins':int(np.count_nonzero(u)),'candidate_bins':int(mask.sum()),
        'maximum_command':float(u.max()),'integrated_base_exposure':float(u.sum()*T),
        'best_Q_found_fine_grid':best_fine,'near_optimum_Q_tolerance':float(tol),
        'time_step_check':{'absolute_Q_difference':float(errQ),'weighted_fraction_RMS_difference':errState,
                           'scheduler_vs_fine_search_Q_difference':float(replay_diff),
                           'allowed_Q_difference':settings.convergence_tolerance},
        'search_dt_used':dt,'material_dt':snapshot.parameters['dt'],
        'recommended_playback_dt':playback_dt,'validation_halvings':validation_halvings,
        'settings':asdict(settings),'warnings':warnings,'budget_reached':budget_hit,
        'elapsed_wall_seconds':time.monotonic()-start_clock}
    notify(f'Done: Q {initial["Q"]:+.6f} -> {final["Q"]:+.6f}; T={T:.6g}. Status: {status}.')
    return DesignResult(snapshot,settings,program,float(T),u.copy(),endpoint,control,trace,mask,qsign,gain,report,scan)
