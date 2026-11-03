"""RF/DNP separation, conservation, peak preservation and budgeted AFP replay."""
from dataclasses import asdict
from pathlib import Path
import copy
import numpy as np
import pytest

from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot, GROUPS, _state_digest, validate_params
from fixture_materials import default_material
from ssrf_realtime.tensor_optimizer import FrozenDynamics, OptimizerSettings, optimize_tensor, replay, synchronized_program
from ssrf_realtime.pulse_program import PulseProgram


def tuned(**kwargs):
    p=dict(n_bins=41,rf_response_mode='independent',rf_pedestal_gain=20.)
    p.update(kwargs)
    return IdealBinModel(IdealBinParams(**p))


def test_preset_preserves_peak_rates_and_accelerates_both_pedestal_transitions():
    # Retain the prior version's empirical preset as an explicit comparator.
    c=MaterialConfig.load(Path(__file__).resolve().parents[1]/'tests/fixtures/materials/pedestal_easy.json')
    m=c.make_model(preserve_state=False)
    old=c.make_model(preserve_state=False); old.params.rf_response_mode='legacy'
    protected=abs(abs(m.Rplus)-m.params.rf_peak_center_R)<=m.params.rf_peak_half_width_R
    np.testing.assert_array_equal(m.rf_rate_weights()[protected],old.rf_rate_weights()[protected])
    for R in (-1.5,1.5):
        j=np.argmin(abs(m.Rplus-R))
        assert m.rf_rate_weights()[j]/old.rf_rate_weights()[j]==pytest.approx(10.)
    np.testing.assert_allclose(m.rf_pedestal_profile(),m.rf_pedestal_profile()[::-1],atol=1e-13)
    np.testing.assert_array_equal(m.capacity_rate_weights(),old.capacity_rate_weights())


def test_peak_only_burn_keeps_entire_trajectory_with_recovery_and_dnp():
    m=tuned(dnp_enabled=True,t1_rate=.03,dt=.0005)
    old=IdealBinModel(copy.deepcopy(m.params)); old.params.rf_response_mode='legacy'
    command=np.zeros(len(m.mu)); command[np.argmin(abs(m.Rplus-.9))]=7.
    program=synchronized_program(m.Rplus,command,.04)
    for model in (m,old):
        model.set_program(program);model.start_program();model.step(80)
    np.testing.assert_array_equal(m.n,old.n)
    np.testing.assert_array_equal(m.total_delivered_exposure,old.total_delivered_exposure)


def test_independent_rf_and_dnp_weights_and_rf_off_dynamics():
    m=tuned(dnp_enabled=True,capacity_rate_power=.7,rf_density_power=.2)
    initial=m.rf_rate_weights().copy()
    m.params.capacity_rate_power=1.8
    np.testing.assert_array_equal(m.rf_rate_weights(),initial)
    dnp=m.capacity_rate_weights().copy()
    off=m.derivative(rf_on=False).copy()
    m.params.rf_pedestal_gain=75.;m.params.rf_density_power=0.
    np.testing.assert_array_equal(m.capacity_rate_weights(),dnp)
    np.testing.assert_array_equal(m.derivative(rf_on=False),off)


@pytest.mark.parametrize('corr',[0.,.4])
def test_modified_dynamics_and_adjoint_match_core_and_finite_difference(corr):
    m=tuned(dnp_enabled=True,capacity_rate_power=.6,rf_density_power=.3,
            t1_rate=.04,orientation_corr_fraction=corr)
    rng=np.random.default_rng(24)
    m.n=m.mu[:,None]*rng.dirichlet([2,3,1],len(m.mu))
    d=FrozenDynamics(PopulationSnapshot.capture(m));u=rng.random(len(m.mu))*3
    m._command_override=u
    rhs=m.derivative(rf_on=True)
    np.testing.assert_allclose(d.rhs(m.n,u),rhs,rtol=2e-12,atol=1e-14)
    np.testing.assert_allclose(rhs.sum(axis=1),0,atol=1e-14)
    a=rng.normal(size=m.n.shape);dn=rng.normal(size=m.n.shape)*m.mu[:,None]
    du=rng.normal(size=len(m.mu));eps=1e-6
    gn,gu=d.vjp(m.n,u,a)
    fd=np.sum(a*(d.rhs(m.n+eps*dn,u+eps*du)-d.rhs(m.n-eps*dn,u-eps*du)))/(2*eps)
    assert fd==pytest.approx(np.sum(gn*dn)+gu@du,abs=1e-8,rel=2e-8)


def test_small_pedestal_exposure_equalizes_both_resonant_pairs_rf_only():
    m=tuned(n_bins=701,diffusion_enabled=False)
    j=np.argmin(abs(m.Rplus-1.5)); k=len(m.mu)-1-j
    u=np.zeros(len(m.mu));u[j]=12.
    s=PopulationSnapshot.capture(m)
    n,_=replay(s,synchronized_program(m.Rplus,u,.2),.2,dt=.00005)
    for a,b,idx in ((0,1,j),(1,2,k)):
        ratio=(n[idx,a]-n[idx,b])/(m.n[idx,a]-m.n[idx,b])
        assert 0 <= ratio < .001
    np.testing.assert_allclose(n.sum(axis=1),m.mu,atol=1e-16)


def test_material_snapshot_migration_and_staleness(tmp_path):
    m=tuned(); c=MaterialConfig.capture(m); raw=c.to_dict()
    assert MaterialConfig.from_dict(raw).parameters==asdict(m.params)
    del raw['parameters']['rf_response']
    legacy=MaterialConfig.from_dict(raw)
    assert legacy.parameters['rf_response_mode']=='legacy'
    snapshot=PopulationSnapshot.capture(m)
    for key in GROUPS['rf_response']: del snapshot.parameters[key]
    snapshot.fingerprint=_state_digest(snapshot.parameters,snapshot.n,snapshot.mu)
    snapshot.save(tmp_path/'old.npz')
    loaded=PopulationSnapshot.load(tmp_path/'old.npz')
    restored=loaded.make_model()
    np.testing.assert_array_equal(restored.n,m.n)
    assert restored.params.rf_response_mode=='legacy' and loaded.matches(restored)
    current=PopulationSnapshot.capture(m)
    m.params.rf_pedestal_gain+=1
    assert not current.matches(m)
    snapshot.parameters['diffusion_scale']+=1
    snapshot.save(tmp_path/'corrupted.npz')
    with pytest.raises(ValueError,match='fingerprint'): PopulationSnapshot.load(tmp_path/'corrupted.npz')


@pytest.mark.parametrize('key,value',[
    ('rf_response_mode','unknown'),('rf_pedestal_gain',0.),('rf_pedestal_gain',float('nan')),
    ('rf_peak_center_R',-1.),('rf_peak_half_width_R',-1.),('rf_transition_width_R',0.),
    ('rf_density_clip',.5),('rf_density_power',-1.)])
def test_invalid_response_rejected(key,value):
    p=asdict(tuned().params);p[key]=value
    with pytest.raises(ValueError,match=key):validate_params(p)


def test_partial_response_group_does_not_silently_migrate():
    raw=MaterialConfig.capture(tuned()).to_dict()
    del raw['parameters']['rf_response']['rf_pedestal_gain']
    with pytest.raises(ValueError):MaterialConfig.from_dict(raw)


def test_budgeted_design_after_afp_with_independent_dnp_replays(tmp_path):
    m=tuned(n_bins=31,dnp_enabled=True,capacity_rate_power=.5,rf_density_power=.8)
    m.afp_sweep([10]);before=m.n.copy()
    s=PopulationSnapshot.capture(m)
    settings=OptimizerSettings(max_power=5,min_duration=.02,max_duration=.05,
        duration_samples=2,duration_refinements=0,max_iterations=15,starts=1,
        max_wall_seconds=60,max_total_exposure=.2)
    result=optimize_tensor(s,settings)
    assert result.report['status']=='completed'
    assert result.report['integrated_base_exposure']<=.2*(1+1e-12)
    np.testing.assert_array_equal(m.n,before)
    result.export(tmp_path/'design')
    restored=PopulationSnapshot.load(tmp_path/'design'/'starting_state.npz')
    program=PulseProgram.load(tmp_path/'design'/'rf_program.json')
    n,_=replay(restored,program,result.duration,dt=result.report['recommended_playback_dt'])
    np.testing.assert_allclose(n,result.final_n,rtol=2e-13,atol=1e-16)
    assert MaterialConfig.load(tmp_path/'design'/'material.json').parameters==s.parameters
