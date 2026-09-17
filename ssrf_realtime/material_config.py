"""Versioned, strict material/settings JSON and independent population snapshots.

All IdealBinParams fields are stored, including backend-only fields. Loading is
transactional. A material file is NOT a state snapshot or a pulse program.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import copy
import hashlib
import json
import math
import os
import tempfile
import numpy as np

from .ideal_model import IdealBinModel, IdealBinParams
from .pulse_program import PulseProgram
from .lineshape import level_populations_from_PQ

SCHEMA = 'ssrf.material.v1'
SNAPSHOT_SCHEMA = 'ssrf.population_snapshot.v1'
GROUPS = {
    'grid': ['n_bins', 'r_min', 'r_max'],
    'lineshape': ['line_gamma', 'line_asym'],
    'initial_state': ['p0', 'q0'],
    'rf_coupling': ['gamma_rf', 'capacity_rate_power', 'capacity_rate_clip'],
    'recovery': ['diffusion_scale','zq_width_R','diffusion_overlap','cross_branch_ratio',
                 'double_quantum_ratio','orientation_corr_fraction','orientation_corr_width_deg',
                 'kernel_cutoff_widths','microwave_diffusion_factor'],
    'dnp': ['p_dnp_sat','dnp_rate'],
    't1': ['t1_rate','t1_p_eq'],
    'run_defaults': ['rf_burn_R','rf_enabled','dnp_enabled','diffusion_enabled'],
    'numerics': ['dt'],
    'display': ['plot_signal_units','plot_divisor','display_scale','calibration_p','noise_sigma'],
    'legacy_voigt_unused_in_ideal_mode': ['rf_gaussian_fwhm_R','rf_lorentzian_fwhm_R',
        'rf_profile_normalization','rf_profile_quadrature_order'],
}


def _finite(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f'{name} must be a finite number')
    return float(x)


def validate_params(data: dict) -> IdealBinParams:
    """Reject unknown/missing keys and invalid values; do not silently clamp."""
    expected = {f.name for f in fields(IdealBinParams)}
    if set(data) != expected:
        raise ValueError(f'Parameter keys differ: missing={sorted(expected-set(data))}; '
                         f'unknown={sorted(set(data)-expected)}')
    defaults = asdict(IdealBinParams())
    for key, value in data.items():
        default = defaults[key]
        if isinstance(default, bool):
            if not isinstance(value, bool): raise ValueError(f'{key} must be true or false')
        elif isinstance(default, int):
            if isinstance(value,bool) or not isinstance(value,int): raise ValueError(f'{key} must be an integer')
        elif isinstance(default, str):
            if not isinstance(value,str): raise ValueError(f'{key} must be a string')
        elif key == 'q0' and value is None:
            pass
        else:
            _finite(value,key)
    p = IdealBinParams(**data)
    if not 5 <= p.n_bins <= 10001: raise ValueError('n_bins must be in [5, 10001]')
    if p.r_min >= p.r_max or not math.isclose(p.r_min,-p.r_max,abs_tol=1e-12,rel_tol=0):
        raise ValueError('Ideal profiles require a finite symmetric increasing R grid')
    if not p.r_min <= p.rf_burn_R <= p.r_max: raise ValueError('Monitor R is outside the grid')
    for name in ('line_gamma','plot_divisor','display_scale','zq_width_R','orientation_corr_width_deg','dt'):
        if getattr(p,name) <= 0: raise ValueError(f'{name} must be positive')
    if p.line_asym >= 3: raise ValueError('line_asym must be less than 3')
    if not 0 <= p.orientation_corr_fraction <= 1: raise ValueError('orientation_corr_fraction must be in [0,1]')
    if p.capacity_rate_clip < 1: raise ValueError('capacity_rate_clip must be >= 1')
    for name in ('gamma_rf','diffusion_scale','cross_branch_ratio','double_quantum_ratio',
                 'kernel_cutoff_widths','microwave_diffusion_factor','capacity_rate_power',
                 'dnp_rate','t1_rate','noise_sigma','rf_gaussian_fwhm_R','rf_lorentzian_fwhm_R'):
        if getattr(p,name) < 0: raise ValueError(f'{name} must be nonnegative')
    for name in ('p0','p_dnp_sat','t1_p_eq','calibration_p'):
        if not -.999999 <= getattr(p,name) <= .999999: raise ValueError(f'{name} must be within +/-0.999999')
    level_populations_from_PQ(p.p0,p.q0)
    if p.diffusion_overlap not in ('lorentzian','gaussian'): raise ValueError('Unknown diffusion_overlap')
    if p.rf_profile_normalization not in ('center_bin','continuous_peak'): raise ValueError('Unknown RF normalization')
    if p.rf_profile_quadrature_order < 0: raise ValueError('RF quadrature order must be nonnegative')
    # These fields are retained for compatibility, but never change ideal RF.
    return p


def write_json(path, data):
    """Atomic replacement after serialization; never leave a partially written file."""
    path = Path(path)
    text = json.dumps(data,indent=2,allow_nan=False,sort_keys=False)+'\n'
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=str(path.parent))
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f: f.write(text)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _read_json(path):
    def fail(token): raise ValueError(f'Invalid JSON numeric constant: {token}')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=fail)


@dataclass
class MaterialConfig:
    name: str
    parameters: dict
    optimizer: dict
    gui: dict
    description: str = ''

    @classmethod
    def capture(cls, model, name='Uncalibrated demonstration', optimizer=None, gui=None, description=''):
        from .tensor_optimizer import OptimizerSettings
        return cls(name,asdict(model.params),asdict(optimizer or OptimizerSettings()),
                   dict(gui or {'steps_per_tick':12,'trace_max_points':4500,'timer_interval_ms':35,
                                'window_width':1000,'window_height':555}),description)

    def validate(self):
        from .tensor_optimizer import OptimizerSettings
        if not isinstance(self.name,str) or not self.name.strip(): raise ValueError('Material name is required')
        if not isinstance(self.description,str): raise ValueError('Description must be text')
        validate_params(self.parameters)
        OptimizerSettings.from_dict(self.optimizer)
        required = {'steps_per_tick','trace_max_points','timer_interval_ms','window_width','window_height'}
        if set(self.gui) != required: raise ValueError('GUI configuration keys do not match schema')
        limits={'steps_per_tick':(1,1000),'trace_max_points':(2,1000000),'timer_interval_ms':(1,10000),
                'window_width':(500,5000),'window_height':(350,5000)}
        for k,(lo,hi) in limits.items():
            v=self.gui[k]
            if isinstance(v,bool) or not isinstance(v,int) or not lo<=v<=hi:
                raise ValueError(f'{k} must be an integer in [{lo},{hi}]')
        return self

    def to_dict(self):
        self.validate()
        if set(sum(GROUPS.values(),[])) != set(self.parameters):
            raise ValueError('Internal parameter grouping is incomplete')
        return {'schema':SCHEMA,'name':self.name,'description':self.description,
                'units':'dimensionless_R__simulation_time__base_RF_rate',
                'parameters':{group:{key:self.parameters[key] for key in keys} for group,keys in GROUPS.items()},
                'optimizer':copy.deepcopy(self.optimizer),'gui':dict(self.gui)}

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or data.get('schema')!=SCHEMA: raise ValueError(f'Expected {SCHEMA}')
        if set(data)-{'schema','name','description','units','parameters','optimizer','gui'}:
            raise ValueError('Unknown top-level material key')
        if data.get('units') != 'dimensionless_R__simulation_time__base_RF_rate': raise ValueError('Unsupported units')
        try:
            nested=data['parameters']
            if set(nested)!=set(GROUPS): raise ValueError('Parameter groups do not match schema')
            flat={}
            for group,keys in GROUPS.items():
                if set(nested[group])!=set(keys): raise ValueError(f'Keys in {group} do not match schema')
                flat.update(nested[group])
            return cls(data['name'],flat,copy.deepcopy(data['optimizer']),dict(data['gui']),
                       data.get('description','')).validate()
        except (KeyError,TypeError,AttributeError) as exc: raise ValueError(f'Malformed material: {exc}') from exc

    def save(self,path): write_json(path,self.to_dict())
    @classmethod
    def load(cls,path): return cls.from_dict(_read_json(path))

    def make_model(self, current=None, preserve_state=True):
        """Build first, commit in GUI later. No regridding of a manipulated state.

        Loading always stops the RF program. Compatible planned programs remain.
        Changed p0/q0 apply on the next explicit reset in preserve-state mode.
        """
        self.validate()
        params=validate_params(self.parameters)
        program=None
        if current is not None and (params.n_bins,params.r_min,params.r_max)==(
            current.params.n_bins,current.params.r_min,current.params.r_max):
            program=current.program
        if preserve_state and current is not None:
            for key in ('n_bins','r_min','r_max','line_gamma','line_asym'):
                if getattr(params,key)!=getattr(current.params,key):
                    raise ValueError(f'{key} changes packet capacities. Choose explicit reinitialize instead; '
                                     'the manipulated state cannot be silently regridded.')
        model=IdealBinModel(params,program)
        if preserve_state and current is not None:
            model.n=current.n.copy(); model.n_ref=current.n_ref.copy(); model.t=float(current.t)
            model.total_delivered_exposure=current.total_delivered_exposure.copy()
        model.stop_program()
        return model


def _state_digest(parameters, n, mu):
    # Exclude controls which cannot change the trajectory of a new ideal program.
    irrelevant={'rf_burn_R','rf_enabled','gamma_rf','p0','q0','noise_sigma','display_scale',
                'plot_signal_units','plot_divisor','calibration_p','rf_gaussian_fwhm_R',
                'rf_lorentzian_fwhm_R','rf_profile_normalization','rf_profile_quadrature_order'}
    kinetics={k:v for k,v in parameters.items() if k not in irrelevant}
    h=hashlib.sha256(json.dumps(kinetics,sort_keys=True,allow_nan=False).encode())
    h.update(np.asarray(n,dtype='<f8').tobytes()); h.update(np.asarray(mu,dtype='<f8').tobytes())
    return h.hexdigest()


@dataclass
class PopulationSnapshot:
    parameters: dict
    n: np.ndarray
    n_ref: np.ndarray
    mu: np.ndarray
    grid: np.ndarray
    simulation_time: float
    fingerprint: str

    @classmethod
    def capture(cls,model):
        p=asdict(model.params)
        return cls(p,model.n.copy(),model.n_ref.copy(),model.mu.copy(),model.Rplus.copy(),
                   float(model.t),_state_digest(p,model.n,model.mu))

    def make_model(self):
        model=IdealBinModel(validate_params(self.parameters))
        if not np.allclose(model.mu,self.mu,rtol=1e-12,atol=1e-15) or not np.array_equal(model.Rplus,self.grid):
            raise ValueError('Snapshot capacities or grid do not agree with its material parameters')
        if self.n.shape!=(len(self.grid),3) or self.n_ref.shape!=self.n.shape:
            raise ValueError('Snapshot population shape is invalid')
        for n in (self.n,self.n_ref):
            if not np.isfinite(n).all() or (n<0).any() or not np.allclose(n.sum(axis=1),self.mu,rtol=1e-10,atol=1e-15):
                raise ValueError('Snapshot populations are not physical')
        if not math.isfinite(self.simulation_time) or self.simulation_time < 0: raise ValueError('Invalid snapshot time')
        if _state_digest(self.parameters,self.n,self.mu)!=self.fingerprint: raise ValueError('Snapshot fingerprint mismatch')
        model.n=self.n.copy(); model.n_ref=self.n_ref.copy(); model.t=self.simulation_time
        model.stop_program()
        return model

    def matches(self,model):
        return self.fingerprint==_state_digest(asdict(model.params),model.n,model.mu)

    def save(self,path):
        metadata={'schema':SNAPSHOT_SCHEMA,'parameters':self.parameters,'simulation_time':self.simulation_time,
                  'fingerprint':self.fingerprint}
        with Path(path).open('wb') as f:
            np.savez_compressed(f,metadata=np.array(json.dumps(metadata,allow_nan=False)),
                                n=self.n,n_ref=self.n_ref,mu=self.mu,grid=self.grid)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'metadata','n','n_ref','mu','grid'}: raise ValueError('Invalid snapshot fields')
            md=json.loads(str(z['metadata'].item()))
            if md.get('schema')!=SNAPSHOT_SCHEMA: raise ValueError('Unknown snapshot schema')
            result=cls(md['parameters'],z['n'].copy(),z['n_ref'].copy(),z['mu'].copy(),z['grid'].copy(),
                       md['simulation_time'],md['fingerprint'])
        result.make_model()
        return result
