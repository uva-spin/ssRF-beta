"""Physical limiting cases, independent kernels, adjoints and budgeted replay."""
from dataclasses import asdict
from pathlib import Path
import copy
import numpy as np
import pytest
from scipy.linalg import expm
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot, GROUPS, _state_digest
from ssrf_realtime.spectral_rf import RFRateOperator
from ssrf_realtime.tensor_optimizer import (FrozenDynamics, OptimizerSettings, candidate_arrays,
                                           optimize_tensor, replay)


def model(**kwargs):
    values=dict(n_bins=51,rf_response_mode='spectral',rf_spectral_strength=.04,
                rf_homogeneous_hwhm_R=.12)
    values.update(kwargs)
    return IdealBinModel(IdealBinParams(**values))


def test_finite_carrier_response_matches_direct_lorentzian_and_transpose():
    m=model(); rng=np.random.default_rng(12); u=rng.random(len(m.mu))
    p=m.params;delta=m.Rplus[:,None]-m.Rplus[None,:]
    matrix=p.rf_spectral_strength*p.rf_homogeneous_hwhm_R/(delta**2+p.rf_homogeneous_hwhm_R**2)
    operator=m.rf_rate_operator();gp,gm=operator.rates(u)
    np.testing.assert_allclose(gp,matrix@u,rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(gm,(matrix@u)[::-1],rtol=1e-13,atol=1e-15)
    a,b=rng.normal(size=(2,len(u)))
    np.testing.assert_allclose(operator.transpose(a,b),matrix.T@(a+b[::-1]),atol=2e-15)
    # A carrier at the edge has the same on-resonance rate; no tail renormalization.
    for j in (0,len(u)//2,len(u)-1):
        u[:]=0;u[j]=1
        gp,_=operator.rates(u)
        assert gp[j]==pytest.approx(p.rf_spectral_strength/p.rf_homogeneous_hwhm_R)
    u[:]=1
    gp,_=operator.rates(u);doubled,_=operator.rates(2*u)
    np.testing.assert_allclose(doubled,2*gp)


def test_linewidth_broadens_response_and_changes_peak_at_fixed_spectral_strength():
    m=model();u=np.zeros(len(m.mu));j=25;u[j]=1
    narrow=m.rf_rate_operator().rates(u)[0]
    assert narrow[j+1]/narrow[j]==pytest.approx(.5)
    m.params.rf_homogeneous_hwhm_R*=2
    broad=m.rf_rate_operator().rates(u)[0]
    assert broad[j]==pytest.approx(narrow[j]/2)
    assert broad[j+2]/broad[j]==pytest.approx(.5)


def test_rf_fractional_dynamics_do_not_depend_on_packet_spin_count_or_old_gain():
    a=model(diffusion_enabled=False);b=model(diffusion_enabled=False,line_gamma=.12,
        rf_density_power=2.,rf_pedestal_gain=100.,capacity_rate_power=0.)
    u=np.linspace(0,1,len(a.mu))
    for m in (a,b):m._command_override=u
    assert not np.allclose(a.mu,b.mu)
    np.testing.assert_allclose(a.derivative(rf_on=True)/a.mu[:,None],
                               b.derivative(rf_on=True)/b.mu[:,None],atol=1e-15)


@pytest.mark.parametrize('q,carrier',[(0.,0.),(2.,2.),(10.,9.)])
def test_full_spin1_dephasing_reference_in_the_weak_drive_limit(q,carrier):
    # Independent 3x3 density matrix, including both RF ladder coherences.
    # Lindblad Iz dephasing gives adjacent-coherence decay Gamma=1.
    omega1=.01;gamma=1.;duration=10000.
    Ix=np.array([[0,1,0],[1,0,1],[0,1,0]])/np.sqrt(2)
    Iz=np.diag([1.,0.,-1.]);L=np.sqrt(2*gamma)*Iz;LL=L@L
    H=np.diag([q-carrier,0.,q+carrier])+omega1*Ix
    def evolve(rho):return -1j*(H@rho-rho@H)+L@rho@L-.5*(LL@rho+rho@LL)
    generator=np.column_stack([evolve(e.reshape(3,3)).ravel() for e in np.eye(9)])
    initial=np.array([.6,.3,.1])
    exact=(expm(duration*generator)@np.diag(initial).ravel()).reshape(3,3)
    gp=omega1**2*gamma/(gamma**2+(q-carrier)**2)
    gm=omega1**2*gamma/(gamma**2+(-q-carrier)**2)
    population=np.array([[-gp,gp,0],[gp,-gp-gm,gm],[0,gm,-gm]])
    predicted=expm(duration*population)@initial
    np.testing.assert_allclose(predicted,np.diag(exact).real,atol=3e-5)


@pytest.mark.parametrize('offset',[0.,1.,3.])
def test_weak_drive_rate_agrees_with_independent_dephasing_density_matrix(offset):
    # Isolated spin-1 ladder transition. In angular-frequency units Gamma=1,
    # H_ab=V, the exact coherence/population system gives W=2 V² Gamma/(Gamma²+delta²).
    gamma=1.;strength=.0002;V=np.sqrt(strength/2);duration=5000.
    bloch=np.array([[-gamma,-offset,0],[offset,-gamma,-2*V],[0,2*V,0.]])
    exact=(expm(duration*bloch)@np.array([0.,0.,1.]))[2]
    op=RFRateOperator(np.array([0.,offset+1e-12]),np.ones(2),strength=strength,hwhm=gamma)
    W=op.rates(np.array([1.,0.]))[0][1]
    rate=np.exp(-2*W*duration)
    assert rate==pytest.approx(exact,rel=.001)


@pytest.mark.parametrize('correlation',[0.,.4])
def test_spectral_adjoint_core_and_candidate_slopes(correlation):
    m=model(dnp_enabled=True,t1_rate=.04,orientation_corr_fraction=correlation)
    rng=np.random.default_rng(7);m.n=m.mu[:,None]*rng.dirichlet([2,3,1],len(m.mu))
    d=FrozenDynamics(PopulationSnapshot.capture(m));u=rng.random(len(m.mu))
    m._command_override=u
    np.testing.assert_allclose(d.rhs(m.n,u),m.derivative(rf_on=True),atol=1e-14,rtol=2e-12)
    adj=rng.normal(size=m.n.shape);dn=rng.normal(size=m.n.shape)*m.mu[:,None];du=rng.normal(size=len(u))
    gn,gu=d.vjp(m.n,u,adj);eps=1e-6
    numeric=np.sum(adj*(d.rhs(m.n+eps*dn,u+eps*du)-d.rhs(m.n-eps*dn,u-eps*du)))/(2*eps)
    assert numeric==pytest.approx(np.sum(gn*dn)+gu@du,abs=1e-9,rel=2e-8)
    _,_,_,gain=candidate_arrays(d,OptimizerSettings())
    u[:]=0;u[23]=1
    difference=d.rhs(m.n,u)-d.rhs(m.n,np.zeros_like(u))
    assert gain[23]==pytest.approx(np.sum(difference@np.array([1.,-2.,1.])),abs=1e-14)
    np.testing.assert_allclose(d.rhs(m.n,u).sum(axis=1),0.,atol=1e-15)


def test_recovery_changes_with_exchange_but_not_rf_saturation_parameters():
    m=model();m.n[20,[0,1]]=np.mean(m.n[20,[0,1]])
    off=m.derivative(rf_on=False)
    m.params.rf_spectral_strength*=4;m.params.rf_homogeneous_hwhm_R*=2
    np.testing.assert_array_equal(m.derivative(rf_on=False),off)
    m.params.diffusion_scale*=2
    np.testing.assert_allclose(m.derivative(rf_on=False),2*off,atol=1e-16)
    assert abs(np.sum(off[:,0]-off[:,2]))<1e-14


def test_new_material_persistence_legacy_snapshot_migration_and_fingerprints(tmp_path):
    m=model();c=MaterialConfig.capture(m)
    assert MaterialConfig.from_dict(c.to_dict()).parameters==asdict(m.params)
    snap=PopulationSnapshot.capture(m)
    m.params.rf_homogeneous_hwhm_R*=2;assert not snap.matches(m)
    old=IdealBinModel(IdealBinParams(n_bins=51,rf_response_mode='independent',rf_pedestal_gain=10))
    snap=PopulationSnapshot.capture(old)
    for key in GROUPS['rf_saturation']:del snap.parameters[key]
    snap.fingerprint=_state_digest(snap.parameters,snap.n,snap.mu);snap.save(tmp_path/'old.npz')
    migrated=PopulationSnapshot.load(tmp_path/'old.npz');restored=migrated.make_model()
    np.testing.assert_array_equal(restored.n,old.n)
    np.testing.assert_array_equal(restored.rf_rate_weights(),old.rf_rate_weights())
    assert migrated.matches(restored)
    snap.parameters['diffusion_scale']+=1;snap.save(tmp_path/'bad.npz')
    with pytest.raises(ValueError,match='fingerprint'):PopulationSnapshot.load(tmp_path/'bad.npz')
    raw=c.to_dict();del raw['parameters']['rf_saturation']['rf_homogeneous_hwhm_R']
    with pytest.raises(ValueError):MaterialConfig.from_dict(raw)


@pytest.mark.parametrize('key,value',[('rf_spectral_strength',-1),('rf_homogeneous_hwhm_R',0),
                                     ('rf_homogeneous_hwhm_R',float('nan'))])
def test_invalid_saturation_parameters_rejected(key,value):
    c=MaterialConfig.capture(model());c.parameters[key]=value
    with pytest.raises(ValueError,match=key):c.validate()


@pytest.mark.parametrize('selectivity',[0.,.7])
def test_spectral_afp_budgeted_design_and_independent_replay(tmp_path,selectivity):
    m=model(n_bins=41,rf_peak_selectivity=selectivity);m.afp_sweep([12]);before=m.n.copy()
    s=PopulationSnapshot.capture(m)
    settings=OptimizerSettings(max_power=5,max_total_exposure=.3,min_duration=.03,max_duration=.08,
        duration_samples=2,duration_refinements=0,starts=1,max_iterations=12,max_wall_seconds=40)
    result=optimize_tensor(s,settings)
    assert result.report['integrated_base_exposure']<=.3*(1+1e-12)
    result.export(tmp_path/'design')
    actual,_=replay(s,result.program,result.duration,dt=result.report['recommended_playback_dt'])
    np.testing.assert_array_equal(actual,result.final_n)
    np.testing.assert_array_equal(m.n,before)
    assert np.min(result.final_n)>=0
    np.testing.assert_allclose(result.final_n.sum(axis=1),m.mu,atol=1e-15)
