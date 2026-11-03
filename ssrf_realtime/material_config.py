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
    'multisite_lineshape': ['line_model', 'line_cd_scale_R', 'line_od_scale_ratio',
                           'line_eta_cd', 'line_eta_od', 'line_od_fraction', 'line_phi_points'],
    'initial_state': ['p0', 'q0'],
    'rf_coupling': ['gamma_rf', 'capacity_rate_power', 'capacity_rate_clip'],
    'rf_response': ['rf_response_mode', 'rf_density_power', 'rf_density_clip',
                    'rf_pedestal_gain', 'rf_peak_center_R', 'rf_peak_half_width_R',
                    'rf_transition_width_R'],
    'rf_saturation': ['rf_spectral_strength', 'rf_homogeneous_hwhm_R'],
    'rf_selectivity': ['rf_peak_selectivity'],
    'afp': ['afp_efficiency', 'afp_subset_indices'],
    'recovery': ['diffusion_scale','zq_width_R','diffusion_overlap','cross_branch_ratio',
                 'double_quantum_ratio','orientation_corr_fraction','orientation_corr_width_deg',
                 'kernel_cutoff_widths','microwave_diffusion_factor'],
    'dnp': ['p_dnp_sat','dnp_rate'],
    't1': ['t1_rate','t1_p_eq'],
    'rf_coupling_profile': ['rf_coupling_density_exponent','rf_coupling_density_clip',
        'rf_coupling_pedestal_gain','rf_coupling_theta_p2','rf_coupling_theta_p4','rf_coupling_od_factor'],
    'rf_width_profile': ['rf_width_density_exponent','rf_width_density_clip',
        'rf_width_theta_p2','rf_width_theta_p4','rf_width_od_factor'],
    'exchange_profile': ['exchange_density_exponent','exchange_density_clip',
        'exchange_theta_p2','exchange_theta_p4','exchange_od_factor'],
    'dnp_profile': ['dnp_theta_p2','dnp_theta_p4','dnp_od_factor'],
    't1_profile': ['t1_density_exponent','t1_density_clip','t1_theta_p2','t1_theta_p4','t1_od_factor'],
    'site_resolved': ['site_resolved','exchange_intersite_factor'],
    'dipolar_reservoir': ['dipolar_enabled','dipolar_capacity_ratio','dipolar_relaxation_rate',
        'dipolar_beta_lattice','dipolar_beta_initial','dipolar_exchange_coupling'],
    'run_defaults': ['rf_burn_R','rf_enabled','dnp_enabled','diffusion_enabled'],
    'numerics': ['dt'],
    'display': ['plot_signal_units','plot_divisor','display_scale','calibration_p','noise_sigma'],
    'legacy_voigt_unused_in_ideal_mode': ['rf_gaussian_fwhm_R','rf_lorentzian_fwhm_R',
        'rf_profile_normalization','rf_profile_quadrature_order'],
}


def _multisite_defaults():
    defaults = asdict(IdealBinParams())
    return {key: defaults[key] for key in GROUPS['multisite_lineshape']}


def _afp_defaults():
    defaults = asdict(IdealBinParams())
    return {key: defaults[key] for key in GROUPS['afp']}


def _rf_response_defaults():
    # Missing group means the original shared RF/DNP density law, never a retune.
    defaults = asdict(IdealBinParams())
    return {key: defaults[key] for key in GROUPS['rf_response']}


def _rf_saturation_defaults():
    defaults = asdict(IdealBinParams())
    return {key: defaults[key] for key in GROUPS['rf_saturation']}


def _rf_selectivity_defaults():
    # Existing spectral materials/snapshots keep MaterialTuning-2 dynamics.
    return {'rf_peak_selectivity': 0.0}


PROFILE_GROUPS = ('rf_coupling_profile','rf_width_profile','exchange_profile','dnp_profile','t1_profile',
                  'site_resolved','dipolar_reservoir')
PROFILE_PREFIXES = ('rf_coupling','rf_width','exchange','dnp','t1')


def _group_defaults(group):
    defaults = asdict(IdealBinParams())
    return {key: defaults[key] for key in GROUPS[group]}


def _orientation_factor_min(a2, a4):
    # min over cos(theta) in [0,1] of 1 + a2*P2 + a4*P4 (dense sampling, exact enough for validation)
    c = np.linspace(0., 1., 2001); c2 = c*c
    return float(np.min(1. + a2*(1.5*c2-.5) + a4*(35*c2*c2-30*c2+3)/8))


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
        if key == 'afp_subset_indices':
            if value is not None and (not isinstance(value, list) or any(
                    isinstance(i, bool) or not isinstance(i, int) for i in value)):
                raise ValueError('afp_subset_indices must be null or a list of integer bin indices')
        elif isinstance(default, bool):
            if not isinstance(value, bool): raise ValueError(f'{key} must be true or false')
        elif isinstance(default, int):
            if isinstance(value,bool) or not isinstance(value,int): raise ValueError(f'{key} must be an integer')
        elif isinstance(default, str):
            if not isinstance(value,str): raise ValueError(f'{key} must be a string')
        elif key == 'q0' and value is None:
            pass
        else:
            _finite(value,key)
    p = IdealBinParams(**copy.deepcopy(data))
    if not 5 <= p.n_bins <= 10001: raise ValueError('n_bins must be in [5, 10001]')
    if not 0 <= p.afp_efficiency <= 1:
        raise ValueError('afp_efficiency must be in [0, 1]')
    seen = set()
    for i in p.afp_subset_indices or []:
        if not 0 <= i < p.n_bins:
            raise ValueError(f'afp_subset_indices must be in [0, {p.n_bins - 1}]')
        if i in seen or p.n_bins - 1 - i in seen:
            raise ValueError('afp_subset_indices must not repeat bins or include both mirror partners')
        seen.add(i)
    if p.r_min >= p.r_max or not math.isclose(p.r_min,-p.r_max,abs_tol=1e-12,rel_tol=0):
        raise ValueError('Ideal profiles require a finite symmetric increasing R grid')
    if not p.r_min <= p.rf_burn_R <= p.r_max: raise ValueError('Monitor R is outside the grid')
    for name in ('line_gamma','plot_divisor','display_scale','zq_width_R','orientation_corr_width_deg','dt'):
        if getattr(p,name) <= 0: raise ValueError(f'{name} must be positive')
    if p.line_asym >= 3: raise ValueError('line_asym must be less than 3')
    if p.line_model not in ('single_pake', 'multisite_butanol'):
        raise ValueError('Unknown line_model')
    if p.line_cd_scale_R <= 0 or p.line_od_scale_ratio <= 1:
        raise ValueError('Multisite splitting must be positive and O-D/C-D ratio > 1')
    for key in ('line_eta_cd', 'line_eta_od', 'line_od_fraction'):
        if not 0 <= getattr(p,key) <= 1: raise ValueError(f'{key} must be in [0,1]')
    if not 8 <= p.line_phi_points <= 512: raise ValueError('line_phi_points must be in [8,512]')
    if p.line_model == 'multisite_butanol' and p.orientation_corr_fraction != 0:
        raise ValueError('Multisite orientation correlation requires site-resolved geometry')
    if not 0 <= p.orientation_corr_fraction <= 1: raise ValueError('orientation_corr_fraction must be in [0,1]')
    if p.capacity_rate_clip < 1: raise ValueError('capacity_rate_clip must be >= 1')
    if p.rf_response_mode not in ('legacy', 'independent', 'spectral'):
        raise ValueError('rf_response_mode must be legacy, independent or spectral')
    if p.rf_homogeneous_hwhm_R <= 0:
        raise ValueError('rf_homogeneous_hwhm_R must be positive')
    if p.rf_spectral_strength < 0:
        raise ValueError('rf_spectral_strength must be nonnegative')
    if not 0 <= p.rf_peak_selectivity <= 1:
        raise ValueError('rf_peak_selectivity must be in [0, 1]')
    if p.rf_density_clip < 1: raise ValueError('rf_density_clip must be >= 1')
    for name in ('rf_pedestal_gain', 'rf_transition_width_R'):
        if getattr(p,name) <= 0: raise ValueError(f'{name} must be positive')
    for name in ('gamma_rf','diffusion_scale','cross_branch_ratio','double_quantum_ratio',
                 'kernel_cutoff_widths','microwave_diffusion_factor','capacity_rate_power',
                 'dnp_rate','t1_rate','noise_sigma','rf_gaussian_fwhm_R','rf_lorentzian_fwhm_R',
                 'rf_density_power','rf_peak_center_R','rf_peak_half_width_R'):
        if getattr(p,name) < 0: raise ValueError(f'{name} must be nonnegative')
    for name in ('p0','p_dnp_sat','t1_p_eq','calibration_p'):
        if not -.999999 <= getattr(p,name) <= .999999: raise ValueError(f'{name} must be within +/-0.999999')
    level_populations_from_PQ(p.p0,p.q0)
    if p.diffusion_overlap not in ('lorentzian','gaussian'): raise ValueError('Unknown diffusion_overlap')
    if p.rf_profile_normalization not in ('center_bin','continuous_peak'): raise ValueError('Unknown RF normalization')
    if p.rf_profile_quadrature_order < 0: raise ValueError('RF quadrature order must be nonnegative')
    for prefix in PROFILE_PREFIXES:
        a2, a4 = getattr(p, prefix+'_theta_p2'), getattr(p, prefix+'_theta_p4')
        if _orientation_factor_min(a2, a4) <= 1e-3:
            raise ValueError(f'{prefix} orientation profile 1+a2*P2+a4*P4 must stay positive for all theta')
        if getattr(p, prefix+'_od_factor') <= 0:
            raise ValueError(f'{prefix}_od_factor must be positive')
        if hasattr(p, prefix+'_density_exponent'):
            if not -4 <= getattr(p, prefix+'_density_exponent') <= 4:
                raise ValueError(f'{prefix}_density_exponent must be in [-4, 4]')
            if getattr(p, prefix+'_density_clip') < 1:
                raise ValueError(f'{prefix}_density_clip must be >= 1')
    if p.rf_coupling_pedestal_gain <= 0: raise ValueError('rf_coupling_pedestal_gain must be positive')
    if p.exchange_intersite_factor < 0: raise ValueError('exchange_intersite_factor must be nonnegative')
    if p.dipolar_capacity_ratio <= 0: raise ValueError('dipolar_capacity_ratio must be positive')
    for name in ('dipolar_relaxation_rate','dipolar_exchange_coupling'):
        if getattr(p,name) < 0: raise ValueError(f'{name} must be nonnegative')
    if p.site_resolved and p.line_model != 'multisite_butanol':
        raise ValueError('site_resolved requires line_model multisite_butanol')
    if p.site_resolved or p.dipolar_enabled:
        if p.rf_response_mode != 'spectral':
            raise ValueError('Site-resolved / dipolar dynamics require rf_response_mode spectral')
        if p.orientation_corr_fraction != 0 or p.kernel_cutoff_widths != 0:
            raise ValueError('Site-resolved / dipolar dynamics require orientation_corr_fraction=0 and kernel_cutoff_widths=0')
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
                                'window_width':1200,'window_height':555}),description)

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
            nested=dict(data['parameters'])
            # Pre-AFP v1 materials omit the entire group. A present group must
            # still contain every field so misspelled/partial settings fail.
            if 'afp' not in nested:
                nested['afp'] = _afp_defaults()
            if 'rf_response' not in nested:
                nested['rf_response'] = _rf_response_defaults()
            if 'rf_saturation' not in nested:
                nested['rf_saturation'] = _rf_saturation_defaults()
            if 'rf_selectivity' not in nested:
                nested['rf_selectivity'] = _rf_selectivity_defaults()
            if 'multisite_lineshape' not in nested:
                nested['multisite_lineshape'] = _multisite_defaults()
            for group in PROFILE_GROUPS:
                if group not in nested:
                    nested[group] = _group_defaults(group)   # older files: uniform profiles (v7 dynamics)
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
            for key in ('n_bins','r_min','r_max','line_gamma','line_asym',*GROUPS['multisite_lineshape']):
                if getattr(params,key)!=getattr(current.params,key):
                    raise ValueError(f'{key} changes packet capacities. Choose explicit reinitialize instead; '
                                     'the manipulated state cannot be silently regridded.')
        model=IdealBinModel(params,program)
        if program is not None:
            # Retain the numerical ceiling when preserving the same planned
            # program. It is runtime metadata and never replaces material dt.
            model.program_integration_dt=current.program_integration_dt
        if preserve_state and current is not None:
            model.n=current.n.copy(); model.n_ref=current.n_ref.copy(); model.t=float(current.t)
            if model.extended_active() and getattr(current,'ext_n',None) is not None:
                try: model.set_extended_state(current.ext_n,current.beta_d)
                except ValueError: model._ensure_ext_state()   # different site split: redistribute totals
            model.total_delivered_exposure=current.total_delivered_exposure.copy()
        model.stop_program()
        return model


def _state_digest(parameters, n, mu, ext_n=None, beta=None):
    # Exclude controls which cannot change the trajectory of a new ideal program.
    irrelevant={'rf_burn_R','rf_enabled','gamma_rf','p0','q0','noise_sigma','display_scale',
                'plot_signal_units','plot_divisor','calibration_p','rf_gaussian_fwhm_R',
                'rf_lorentzian_fwhm_R','rf_profile_normalization','rf_profile_quadrature_order'}
    kinetics={k:v for k,v in parameters.items() if k not in irrelevant}
    h=hashlib.sha256(json.dumps(kinetics,sort_keys=True,allow_nan=False).encode())
    h.update(np.asarray(n,dtype='<f8').tobytes()); h.update(np.asarray(mu,dtype='<f8').tobytes())
    if ext_n is not None:   # V8 site-resolved populations and dipolar inverse temperature
        h.update(b'ext'); h.update(np.asarray(ext_n,dtype='<f8').tobytes())
        h.update(np.asarray([beta],dtype='<f8').tobytes())
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
    ext_n: Optional[np.ndarray] = None
    beta: float = 0.0

    @classmethod
    def capture(cls,model):
        p=asdict(model.params)
        ext=None; beta=0.0
        if getattr(model,'extended_active',lambda:False)():
            model._ensure_ext_state(); ext=model.ext_n.copy(); beta=float(model.beta_d)
        return cls(p,model.n.copy(),model.n_ref.copy(),model.mu.copy(),model.Rplus.copy(),
                   float(model.t),_state_digest(p,model.n,model.mu,ext,beta),ext,beta)

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
        if _state_digest(self.parameters,self.n,self.mu,self.ext_n,self.beta)!=self.fingerprint:
            raise ValueError('Snapshot fingerprint mismatch')
        model.n=self.n.copy(); model.n_ref=self.n_ref.copy(); model.t=self.simulation_time
        if self.ext_n is not None:
            if not model.extended_active(): raise ValueError('Snapshot has extended state but material is not extended')
            model.set_extended_state(self.ext_n,self.beta)
            if not np.allclose(model.n,self.n,rtol=1e-12,atol=1e-15): raise ValueError('Extended state disagrees with totals')
            model.n=self.n.copy(); model._ext_collapsed=model.n.copy()
        model.stop_program()
        return model

    def matches(self,model):
        ext=None; beta=None
        if getattr(model,'extended_active',lambda:False)():
            model._ensure_ext_state(); ext=model.ext_n; beta=float(model.beta_d)
        return self.fingerprint==_state_digest(asdict(model.params),model.n,model.mu,ext,beta)

    def save(self,path):
        metadata={'schema':SNAPSHOT_SCHEMA,'parameters':self.parameters,'simulation_time':self.simulation_time,
                  'fingerprint':self.fingerprint}
        extra={} if self.ext_n is None else {'ext_n':self.ext_n,'beta':np.array([self.beta])}
        with Path(path).open('wb') as f:
            np.savez_compressed(f,metadata=np.array(json.dumps(metadata,allow_nan=False)),
                                n=self.n,n_ref=self.n_ref,mu=self.mu,grid=self.grid,**extra)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            base={'metadata','n','n_ref','mu','grid'}
            if set(z.files) not in (base, base|{'ext_n','beta'}): raise ValueError('Invalid snapshot fields')
            md=json.loads(str(z['metadata'].item()))
            if md.get('schema')!=SNAPSHOT_SCHEMA: raise ValueError('Unknown snapshot schema')
            ext=z['ext_n'].copy() if 'ext_n' in z.files else None
            beta=float(z['beta'][0]) if 'beta' in z.files else 0.0
            result=cls(md['parameters'],z['n'].copy(),z['n_ref'].copy(),z['mu'].copy(),z['grid'].copy(),
                       md['simulation_time'],md['fingerprint'],ext,beta)
        if not isinstance(result.parameters, dict):
            raise ValueError('Snapshot parameters must be an object')
        additions = {}
        for group,defaults in (('afp', _afp_defaults), ('rf_response', _rf_response_defaults),
                               ('rf_saturation', _rf_saturation_defaults),
                               ('rf_selectivity', _rf_selectivity_defaults),
                               ('multisite_lineshape', _multisite_defaults),
                               *[(g, (lambda g=g: _group_defaults(g))) for g in PROFILE_GROUPS]):
            if not any(key in result.parameters for key in GROUPS[group]):
                additions.update(defaults())
        if additions:
            # Authenticate the old payload before adding defaults. Recompute
            # the fingerprint afterward so matches() works with restored models
            # and subsequent exports carry a consistent current parameter set.
            if _state_digest(result.parameters,result.n,result.mu) != result.fingerprint:
                raise ValueError('Snapshot fingerprint mismatch')
            result.parameters = {**result.parameters, **additions}
            result.fingerprint = _state_digest(result.parameters,result.n,result.mu)
        result.make_model()
        return result
