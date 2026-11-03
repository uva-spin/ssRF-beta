"""Selectivity, mirror mapping, inherited kinetics and old-file compatibility."""
from dataclasses import asdict
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from fixture_materials import default_material
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot, _state_digest
from ssrf_realtime.tensor_optimizer import FrozenDynamics


def make_model(selectivity=.7, bins=81):
    config = default_material()
    config.parameters.update(rf_peak_selectivity=selectivity, n_bins=bins)
    return config.make_model(preserve_state=False)


def test_selectivity_preserves_outer_pedestal_and_strongest_packet():
    model = make_model(bins=701)
    response = model.rf_packet_response()
    peak = np.argmax(model.mu)
    assert response[peak] == 1.
    assert 0.3 < response[len(response)-1-peak] < .35
    distance = abs(abs(model.Rplus)-model.params.rf_peak_center_R)
    outer = distance >= model.params.rf_peak_half_width_R+model.params.rf_transition_width_R
    np.testing.assert_array_equal(response[outer], 1.)
    assert np.all((response > 0) & (response <= 1))
    model.params.rf_peak_selectivity = 0
    np.testing.assert_array_equal(model.rf_packet_response(), np.ones(len(response)))


def test_packet_weighted_kernel_and_transpose_use_correct_mirror_order():
    model = make_model();p=model.params;R=model.Rplus
    K=p.rf_spectral_strength*p.rf_homogeneous_hwhm_R/(
        p.rf_homogeneous_hwhm_R**2+(R[:,None]-R[None,:])**2)
    c=model.rf_packet_response();rng=np.random.default_rng(21)
    u,a,b=rng.normal(size=(3,len(R)))
    operator=model.rf_rate_operator();gp,gm=operator.rates(u)
    np.testing.assert_allclose(gp,c*(K@u),atol=2e-14)
    np.testing.assert_allclose(gm,c*(K@u)[::-1],atol=2e-14)
    np.testing.assert_allclose(operator.transpose(a,b),K.T@(c*a+(c*b)[::-1]),atol=2e-14)
    assert not np.allclose(gm,gp[::-1])
    model.params.rf_peak_selectivity=.2
    assert model.rf_rate_operator() is not operator


@pytest.mark.parametrize('correlation',[0.,.4])
def test_selective_live_kinetics_and_adjoint_match_directional_derivative(correlation):
    model=make_model(bins=61)
    model.params.orientation_corr_fraction=correlation
    model.params.dnp_enabled=True;model.params.t1_rate=.03
    rng=np.random.default_rng(13)
    model.n=model.mu[:,None]*rng.dirichlet([2,3,1],len(model.mu))
    d=FrozenDynamics(PopulationSnapshot.capture(model));u=rng.random(d.N)
    model._command_override=u
    np.testing.assert_allclose(d.rhs(model.n,u),model.derivative(rf_on=True),atol=1e-14)
    adj=rng.normal(size=model.n.shape);dn=rng.normal(size=model.n.shape)*d.mu[:,None]
    du=rng.normal(size=d.N);gn,gu=d.vjp(model.n,u,adj);eps=1e-6
    numeric=np.sum(adj*(d.rhs(model.n+eps*dn,u+eps*du)-d.rhs(model.n-eps*dn,u-eps*du)))/(2*eps)
    assert numeric==pytest.approx(np.sum(gn*dn)+gu@du,abs=2e-9,rel=2e-8)
    Q,n,grad=d.evaluate(u,.08,.001,gradient=True)
    qp,_=d.evaluate(u+eps*du,.08,.001);qm,_=d.evaluate(u-eps*du,.08,.001)
    assert (qp-qm)/(2*eps)==pytest.approx(grad@du,abs=2e-9,rel=2e-7)
    np.testing.assert_allclose(n.sum(axis=1),d.mu,atol=1e-15)


def test_selectivity_does_not_change_recovery_dnp_or_packet_populations():
    model=make_model();model.params.dnp_enabled=True
    model.n[20,[0,1]]=np.mean(model.n[20,[0,1]])
    state=model.n.copy();off=model.derivative(rf_on=False)
    model.params.rf_peak_selectivity=0
    np.testing.assert_array_equal(model.derivative(rf_on=False),off)
    np.testing.assert_array_equal(model.n,state)


def test_materialtuning2_files_migrate_without_adopting_new_selectivity(tmp_path):
    model=make_model(selectivity=0.)
    config=MaterialConfig.capture(model);raw=config.to_dict()
    del raw['parameters']['rf_selectivity']
    loaded=MaterialConfig.from_dict(raw)
    assert loaded.parameters==asdict(model.params)
    snapshot=PopulationSnapshot.capture(model)
    del snapshot.parameters['rf_peak_selectivity']
    snapshot.fingerprint=_state_digest(snapshot.parameters,snapshot.n,snapshot.mu)
    snapshot.save(tmp_path/'v2.npz')
    migrated=PopulationSnapshot.load(tmp_path/'v2.npz')
    assert migrated.matches(model)
    u=np.linspace(0,1,len(model.mu))
    for old,new in zip(model.rf_rate_operator().rates(u),migrated.make_model().rf_rate_operator().rates(u)):
        np.testing.assert_array_equal(old,new)
    snapshot.n[0,0]+=1e-6;snapshot.save(tmp_path/'tampered.npz')
    with pytest.raises(ValueError,match='fingerprint'):PopulationSnapshot.load(tmp_path/'tampered.npz')


@pytest.mark.parametrize('value',[-.1,1.1,float('nan'),True])
def test_invalid_selectivity_is_rejected(value):
    c=default_material();c.parameters['rf_peak_selectivity']=value
    with pytest.raises(ValueError,match='rf_peak_selectivity'):c.validate()


def test_broad_burn_restores_crossing_without_changing_diffusion():
    ratios=[]
    for alpha in (0.,.7):
        m=make_model(alpha,bins=701);d=FrozenDynamics(PopulationSnapshot.capture(m));R=m.Rplus
        u=.08*((R>=-1.01)&(R<=-.6))+.06*((R>=1.13)&(R<=2.3))
        sol=solve_ivp(lambda t,y:d.rhs(y.reshape(d.N,3),u).ravel(),(0,2),d.n0.ravel(),
                      method='DOP853',rtol=2e-9,atol=np.repeat(d.mu*1e-12,3))
        assert sol.success
        n=sol.y[:,-1].reshape(d.N,3);j=np.argmax(m.spectrum()[2]);k=d.N-1-j
        ratios.append((n[k,1]-n[k,2])/(n[j,0]-n[j,1]))
        assert np.min(n)>=0
        np.testing.assert_allclose(n.sum(axis=1),d.mu,atol=1e-14)
    assert ratios[0]>2
    assert ratios[1]<1
