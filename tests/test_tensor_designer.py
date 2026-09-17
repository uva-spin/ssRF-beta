"""New workflow regression tests; baseline kinetics tests remain unedited."""
import json
from dataclasses import asdict, fields
from threading import Event
import numpy as np
import pytest

from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.pulse_program import PulseProgram
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot, validate_params, GROUPS
from ssrf_realtime.tensor_optimizer import (OptimizerSettings, FrozenDynamics, candidate_arrays,
    optimize_tensor, replay, polarizations, synchronized_program, OptimizationCancelled)


def small(**kw): return IdealBinModel(IdealBinParams(n_bins=31,**kw))
def settings(**kw):
    d=dict(max_power=12.,min_duration=.02,max_duration=.18,duration_samples=3,
           duration_refinements=1,max_iterations=12,starts=1,max_wall_seconds=60.)
    d.update(kw);return OptimizerSettings(**d)


def test_all_parameters_are_saved(tmp_path):
    m=small(); c=MaterialConfig.capture(m,optimizer=settings())
    d=c.to_dict();flat={k:v for g in d['parameters'].values() for k,v in g.items()}
    assert flat==asdict(m.params)
    assert set(sum(GROUPS.values(),[]))=={f.name for f in fields(IdealBinParams)}
    c.save(tmp_path/'m.json');assert MaterialConfig.load(tmp_path/'m.json').to_dict()==d


def test_material_preserves_manipulated_state_without_renormalization(tmp_path):
    m=small();m.start_program();m.step(20)
    n=m.n.copy();ref=m.n_ref.copy();t=m.t
    c=MaterialConfig.capture(m,optimizer=settings());c.parameters['diffusion_scale']=13.
    c.parameters['cross_branch_ratio']=.33;c.parameters['p0']=-.58
    other=c.make_model(m,True)
    np.testing.assert_array_equal(other.n,n);np.testing.assert_array_equal(other.n_ref,ref)
    assert other.t==t and not other.params.rf_enabled
    assert other.params.diffusion_scale==13 and other.params.p0==-.58
    assert other.program_state=='stopped'
    other.reset();assert polarizations(other.n)['P']==pytest.approx(-.58)
    np.testing.assert_array_equal(m.n,n)


@pytest.mark.parametrize('key,value',[('n_bins',33),('line_gamma',.08),('line_asym',.02)])
def test_material_refuses_silent_capacity_change(key,value):
    m=small();n=m.n.copy();c=MaterialConfig.capture(m);c.parameters[key]=value
    with pytest.raises(ValueError,match='capacities'):c.make_model(m,True)
    np.testing.assert_array_equal(m.n,n)
    other=c.make_model(m,False);assert getattr(other.params,key)==value


@pytest.mark.parametrize('key,value',[('diffusion_scale',-1.),('p0',1.5),('dt',0.),
    ('n_bins',31.5),('diffusion_enabled',1),('noise_sigma',float('nan')),('q0',2.),
    ('diffusion_overlap','made_up'),('capacity_rate_clip',.5)])
def test_invalid_material_rejected(key,value):
    d=asdict(IdealBinParams(n_bins=31));d[key]=value
    with pytest.raises(ValueError):validate_params(d)


def test_unknown_material_field_rejected():
    c=MaterialConfig.capture(small()).to_dict();c['parameters']['recovery']['mystery']=3
    with pytest.raises(ValueError):MaterialConfig.from_dict(c)


def test_snapshot_roundtrip_and_staleness(tmp_path):
    m=small();m.start_program();m.step(30);s=PopulationSnapshot.capture(m)
    s.save(tmp_path/'s.npz');s2=PopulationSnapshot.load(tmp_path/'s.npz')
    np.testing.assert_array_equal(s2.make_model().n,m.n)
    assert s2.matches(m)
    m.params.rf_burn_R=.8;m.params.noise_sigma=.01
    assert s2.matches(m)
    m.params.dnp_enabled=not m.params.dnp_enabled
    assert not s2.matches(m)
    s2.n[0,0]*=2
    with pytest.raises(ValueError):s2.make_model()


@pytest.mark.parametrize('corr',[0.,.3,1.])
@pytest.mark.parametrize('overlap',['gaussian','lorentzian'])
@pytest.mark.parametrize('cutoff',[0.,4.])
def test_forward_regrouping_is_same_as_core(corr,overlap,cutoff):
    m=small(orientation_corr_fraction=corr,diffusion_overlap=overlap,kernel_cutoff_widths=cutoff,
            dnp_enabled=True,dnp_rate=.17,p_dnp_sat=-.3,t1_rate=.07,t1_p_eq=.1,
            double_quantum_ratio=.27,cross_branch_ratio=.8,microwave_diffusion_factor=1.4)
    rng=np.random.default_rng(8)
    m.n=m.mu[:,None]*rng.dirichlet([2.,1.,3.],len(m.mu))
    s=PopulationSnapshot.capture(m);d=FrozenDynamics(s)
    u=rng.random(len(m.mu))*5
    m._command_override=u
    np.testing.assert_allclose(d.rhs(m.n,u),m.derivative(rf_on=True),rtol=2e-12,atol=2e-15)
    assert np.max(abs(d.rhs(m.n,u).sum(axis=1)))<1e-14


@pytest.mark.parametrize('corr',[0.,.25])
def test_analytic_vector_jacobian_product(corr):
    m=small(dnp_enabled=True,t1_rate=.1,orientation_corr_fraction=corr)
    rng=np.random.default_rng(12);m.n=m.mu[:,None]*rng.dirichlet([3.,2.,1.],len(m.mu))
    d=FrozenDynamics(PopulationSnapshot.capture(m));u=rng.random(len(m.mu))*2
    a=rng.normal(size=m.n.shape);dn=rng.normal(size=m.n.shape)*m.mu[:,None];du=rng.normal(size=len(m.mu))
    gn,gu=d.vjp(m.n,u,a);eps=1e-6
    fd=np.sum(a*(d.rhs(m.n+eps*dn,u+eps*du)-d.rhs(m.n-eps*dn,u-eps*du)))/(2*eps)
    assert fd==pytest.approx(np.sum(gn*dn)+gu@du,abs=1e-9,rel=2e-8)


def test_discrete_endpoint_adjoint_matches_finite_difference():
    m=small(dnp_enabled=True,t1_rate=.02);d=FrozenDynamics(PopulationSnapshot.capture(m))
    rng=np.random.default_rng(2);u=rng.random(len(m.mu))*2;du=rng.normal(size=len(m.mu))
    Q,n,g=d.evaluate(u,.08,.002,gradient=True)
    eps=1e-5
    q1=d.evaluate(u+eps*du,.08,.002)[0];q2=d.evaluate(u-eps*du,.08,.002)[0]
    assert (q1-q2)/(2*eps)==pytest.approx(g@du,abs=2e-9,rel=2e-6)


def test_negative_P_uses_signed_geometry_not_reversed_inequality():
    a=FrozenDynamics(PopulationSnapshot.capture(small(p0=.45)))
    b=FrozenDynamics(PopulationSnapshot.capture(small(p0=-.45)))
    ma,_,qa,ga=candidate_arrays(a,settings());mb,_,qb,gb=candidate_arrays(b,settings())
    np.testing.assert_array_equal(ma,mb[::-1]);np.testing.assert_allclose(qa,qb[::-1],atol=1e-14)
    np.testing.assert_allclose(ga,gb[::-1],atol=1e-14)


def test_user_regions_and_per_bin_ceilings():
    d=FrozenDynamics(PopulationSnapshot.capture(small()))
    o=settings(regions=[[-1.,-.5]],per_bin_power_limits={'10':0.,'11':2.})
    mask,upper,*_=candidate_arrays(d,o)
    assert np.all(upper[(d.snapshot.grid<-1)|(d.snapshot.grid>-.5)]==0)
    assert upper[10]==0
    assert upper[11]<=2


@pytest.fixture(scope='module')
def result():
    m=small();n=m.n.copy()
    # Deliberately non-Boltzmann starting state, not regenerated from P.
    j=12; amount=m.n[j,0]*.08;m.n[j,0]-=amount;m.n[j,1]+=amount
    s=PopulationSnapshot.capture(m);r=optimize_tensor(s,settings())
    np.testing.assert_array_equal(s.n,m.n)
    assert not np.array_equal(m.n,n)
    return r


def test_synchronized_design_is_bounded_and_improves_Q(result):
    r=result
    assert r.report['status']=='completed'
    assert r.report['final']['Q']>r.report['initial']['Q']
    assert r.report['final']['P']<r.report['initial']['P']
    assert r.powers.max()<=r.settings.max_power
    rows=[p for profile in r.program.profiles for p in profile.pulses if p.enabled]
    assert all(p.start==0 and p.duration==r.duration for p in rows)
    assert r.program.gain==1.
    assert np.all(r.powers[~r.candidates]==0)
    np.testing.assert_allclose(r.program.compile().exposure,r.powers*r.duration,atol=1e-14)
    assert r.report['time_step_check']['absolute_Q_difference']<=r.settings.convergence_tolerance


def test_export_roundtrip_replay(result,tmp_path):
    r=result;r.export(tmp_path)
    a=PulseProgram.load(tmp_path/'rf_program.json');b=PulseProgram.load_csv(tmp_path/'rf_program.csv')
    assert a.to_dict()==b.to_dict()
    state=PopulationSnapshot.load(tmp_path/'starting_state.npz')
    n,_=replay(state,b,r.duration,dt=r.report['recommended_playback_dt'])
    np.testing.assert_allclose(n,r.final_n,rtol=2e-13,atol=1e-16)
    assert MaterialConfig.load(tmp_path/'material.json').parameters==r.snapshot.parameters
    report=json.loads((tmp_path/'optimization_report.json').read_text())
    assert report['snapshot_fingerprint']==state.fingerprint
    assert (tmp_path/'bin_diagnostics.csv').exists()


def test_high_tensor_state_not_forced_to_accept_damaging_RF():
    s=PopulationSnapshot.capture(small(p0=.2,q0=1.,diffusion_enabled=False))
    r=optimize_tensor(s,settings(candidate_mode='all',duration_samples=2,duration_refinements=0))
    assert r.report['status']=='no_improvement'
    assert r.duration==0 and not r.powers.any()
    np.testing.assert_array_equal(r.final_n,s.n)


def test_negative_polarization_design():
    r=optimize_tensor(PopulationSnapshot.capture(small(p0=-.45)),
                      settings(duration_samples=2,duration_refinements=0))
    assert r.report['final']['Q']>r.report['initial']['Q']
    assert abs(r.report['final']['P'])<abs(r.report['initial']['P'])


def test_cancellation_does_not_touch_live_model():
    m=small();n=m.n.copy();event=Event();event.set()
    with pytest.raises(OptimizationCancelled):optimize_tensor(PopulationSnapshot.capture(m),settings(),cancel=event)
    np.testing.assert_array_equal(m.n,n)


@pytest.mark.parametrize('key,value',[('max_power',0),('max_duration',-.1),('duration_samples',1),
                                    ('candidate_mode','nonsense'),('max_wall_seconds',float('inf')),
                                    ('per_bin_power_limits',{'1':-3}),('regions',[[2,1]])])
def test_bad_optimizer_settings(key,value):
    o=settings();setattr(o,key,value)
    with pytest.raises(ValueError):o.validate()


def test_no_display_noise_or_calibration_in_candidate_calculation():
    a=small(display_scale=1.,noise_sigma=0.)
    b=small(display_scale=4.,noise_sigma=.02)
    da=FrozenDynamics(PopulationSnapshot.capture(a));db=FrozenDynamics(PopulationSnapshot.capture(b))
    for x,y in zip(candidate_arrays(da,settings()),candidate_arrays(db,settings())):
        np.testing.assert_array_equal(x,y)


def test_snapshot_does_not_confuse_same_total_P_with_same_signal():
    m=small();s=PopulationSnapshot.capture(m)
    d=min(m.n[10,0],m.n[11,1])*.02
    m.n[10,0]-=d;m.n[10,1]+=d;m.n[11,0]+=d;m.n[11,1]-=d
    assert polarizations(m.n)['P']==pytest.approx(polarizations(s.n)['P'])
    assert not s.matches(m)


def test_single_bin_optimizer_matches_analytic_RF_exposure():
    m=IdealBinModel(IdealBinParams(n_bins=101,diffusion_enabled=False,dt=.0001))
    j=int(np.argmin(abs(m.Rplus+.94)))
    s=PopulationSnapshot.capture(m);d=FrozenDynamics(s)
    R=float(m.Rplus[j]);T=.2
    cp=d.w[j];cm=d.w[::-1][j]
    a=m.n[j,0]-m.n[j,1];b=(m.n[:,1]-m.n[:,2])[::-1][j]
    E=np.log(cm*b/(cp*a))/(2*(cm-cp))
    assert E>0
    expected=E/T
    r=optimize_tensor(s,settings(min_duration=T,max_duration=T,duration_samples=2,duration_refinements=0,
        search_dt=.0001,max_power=20,regions=[[R,R]],max_iterations=30))
    assert r.powers[j]==pytest.approx(expected,rel=.003)
    assert np.count_nonzero(r.powers)==1
