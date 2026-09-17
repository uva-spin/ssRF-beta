"""Recovery tests separate thermodynamic correctness from material calibration."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import pytest
from ssrf_realtime.model import Spin1Model, Spin1Params
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.lineshape import level_populations_from_PQ
from ssrf_realtime.pulse_program import BinPulse, RFProfile, PulseProgram


def test_preserved_RF_spectra_DNP_utilities_and_scheduler_methods():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'RECOVERY_SOURCE_MANIFEST.json').read_text())
    for filename,digest in manifest['unchanged_python'].items():
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==digest
    for key,digest in manifest['preserved_methods'].items():
        filename,qualified=key.split(':'); cls,name=qualified.split('.')
        tree=ast.parse((root/filename).read_text())
        c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
        method=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name)
        assert hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest()==digest


@pytest.mark.parametrize('P',[-.8,-.45,-.1,0,.1,.45,.8])
def test_each_recovery_channel_is_stationary_at_Boltzmann(P):
    m=Spin1Model(Spin1Params(n_bins=101,p0=P,zq_width_R=.10))
    for name,d in m._spin_diffusion_terms(False).items():
        assert np.max(abs(d))<3e-16, name
    assert m.recovery_equilibrium_diagnostics()['fraction_rms_error']<1e-14


def test_non_Boltzmann_uniform_state_is_no_longer_a_stationary_state():
    m=Spin1Model(Spin1Params(n_bins=101,p0=.45,q0=.3,zq_width_R=.1))
    terms=m._spin_diffusion_terms(False)
    # A uniformly different Q has no same-branch gradients, but now equilibrates.
    assert np.max(abs(terms['diff_plus0']))<1e-16
    assert np.max(abs(terms['diff_0minus']))<1e-16
    assert np.max(abs(terms['diff_double_quantum']))<1e-16
    dn=sum(terms.values())
    assert abs(np.sum(dn[:,0]-dn[:,2]))<1e-14
    assert np.sum(dn[:,0]-2*dn[:,1]+dn[:,2])<0


@pytest.mark.parametrize('correlation',[0,.4,1])
def test_each_channel_preserves_packet_mass_vector_area_and_has_positive_entropy_production(correlation):
    m=Spin1Model(Spin1Params(n_bins=81,orientation_corr_fraction=correlation,zq_width_R=.15))
    rng=np.random.default_rng(672)
    p=rng.uniform(.01,1,size=m.n.shape);p/=p.sum(axis=1,keepdims=True)
    m.n=m.mu[:,None]*p
    for name,d in m._spin_diffusion_terms(False).items():
        assert np.max(abs(d.sum(axis=1)))<3e-16,name
        assert abs(np.sum(d[:,0]-d[:,2]))<3e-15,name
        entropy_dot=-np.sum(d*np.log(p))
        assert entropy_dot>=-3e-14,(name,entropy_dot)
    assert abs(np.sum(m._spin_diffusion_terms(False)['diff_cross'][:,1]))>1e-8


def test_DQ_does_not_change_Q_and_matches_explicit_pair_counting():
    m=Spin1Model(Spin1Params(n_bins=21,zq_width_R=.4))
    rng=np.random.default_rng(70)
    p=rng.random(m.n.shape);p/=p.sum(axis=1,keepdims=True);m.n=m.mu[:,None]*p
    d=m._double_quantum_term(5)
    expected=np.zeros_like(d)
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            flow=.5*m.mu[i]*m.mu[j]*(p[i,2]*p[j,0]-p[i,0]*p[j,2])
            expected[i,0]+=flow;expected[i,2]-=flow
            expected[j,0]-=flow;expected[j,2]+=flow
    np.testing.assert_allclose(d,expected,rtol=2e-14,atol=2e-17)
    np.testing.assert_array_equal(d[:,1],np.zeros(len(d)))
    assert abs(np.sum(d[:,0]-2*d[:,1]+d[:,2]))<1e-16


def test_recovery_diagnostics_are_read_only_and_warn_about_restricted_channels():
    m=Spin1Model(Spin1Params(cross_branch_ratio=0,double_quantum_ratio=0))
    before=m.n.copy();d=m.recovery_equilibrium_diagnostics()
    np.testing.assert_array_equal(before,m.n)
    assert any('impossible' in x for x in d['warnings'])
    m.params.zq_width_R=.001
    m.params.diffusion_overlap='gaussian';m.params.kernel_cutoff_widths=1
    assert not m.recovery_equilibrium_diagnostics()['connected_same_branch_graph']


def test_default_transport_graph_is_connected_and_tensor_exchange_on():
    m=Spin1Model()
    d=m.recovery_equilibrium_diagnostics()
    assert d['connected_same_branch_graph']
    assert not d['warnings']
    assert m.params.cross_branch_ratio>0 and m.params.double_quantum_ratio>0


def test_RF_remains_an_uncancelled_sink_during_simultaneous_recovery():
    m=IdealBinModel(IdealBinParams(n_bins=101,dt=.002,p0=.45))
    k=m.branch_indices(.42)[0]
    program=PulseProgram(len(m.Rplus),m.Rplus[0],m.Rplus[-1],
                         profiles=[RFProfile('test',pulses=[BinPulse(k,1.0,0,.2)])])
    m.set_program(program);m.start_program()
    Pstart=m.polarizations()['P'];ledger=0.
    for _ in range(60):
        _,parts=m.derivative(breakdown=True)
        diff=sum(parts[x]['dP_dt'] for x in ('diff_plus0','diff_0minus','diff_cross','diff_double_quantum'))
        assert abs(diff)<1e-13
        assert parts['RF']['dP_dt']<0
        ledger+=m.params.dt*parts['RF']['dP_dt']
        m.step()
    assert m.polarizations()['P']<Pstart
    assert abs(m.polarizations()['P']-Pstart-ledger)<2e-13


def test_high_new_rates_are_subdivided_without_pulse_time_or_power_clipping():
    m=IdealBinModel(IdealBinParams(n_bins=41,dt=.05,zq_width_R=.2,
                                  diffusion_scale=50,double_quantum_ratio=4))
    m.n[20]=m.mu[20]*np.array([.2,.7,.1])
    initialP=m.polarizations()['P'];m.step(2,rf_on=False,dnp_on=False)
    assert m.last_substeps>1
    assert m.t==pytest.approx(.1)
    assert np.min(m.n)>=0
    assert m.polarizations()['P']==pytest.approx(initialP,abs=3e-14)


def _prepare_RF_only(m,R,exposure=.3):
    # Exact pair equalization prepares a state; it is not a recovery step.
    kp,km=m.branch_indices(R)
    assert kp!=km
    for k,a,b in [(kp,0,1),(km,1,2)]:
        avg=.5*(m.n[k,a]+m.n[k,b])
        delta=(m.n[k,a]-m.n[k,b])*np.exp(-2*exposure*m.capacity_rate_weights()[k])
        m.n[k,a]=avg+delta/2;m.n[k,b]=avg-delta/2
    return kp,km


@pytest.mark.parametrize('R',[-1.5,-.94,-.4,.4,.94,1.5])
def test_large_mirror_inversion_is_absent_in_default_recovery_benchmarks(R):
    scipy=pytest.importorskip('scipy.integrate')
    m=Spin1Model(Spin1Params(n_bins=181,dt=.005))
    kp,km=_prepare_RF_only(m,R,exposure=.3)
    N=len(m.mu);n0=m.n.copy();P=m.polarizations()['P']
    def rhs(t,y):
        m.n=y.reshape(N,3)
        return sum(m._spin_diffusion_terms(False).values()).ravel()
    times=np.r_[0,np.geomspace(.01,100,90)]
    sol=scipy.solve_ivp(rhs,(0,100),n0.ravel(),t_eval=times,method='DOP853',rtol=2e-9,atol=1e-13)
    assert sol.success
    n=sol.y.T.reshape(-1,N,3);p=n/m.mu[None,:,None]
    for k,diff in [(km,p[:,:,0]-p[:,:,1]),(kp,p[:,:,1]-p[:,:,2])]:
        excess=diff[:,k]-.5*(diff[:,k-1]+diff[:,k+1])
        assert excess[0]>0
        # A two-neighbor difference also samples curvature of a smooth,
        # redistributed background. Bound any negative local contrast to 0.01%
        # of its initial peak, rather than claiming universal pointwise
        # monotonicity. The old model produced order-10% narrow local deficits.
        assert excess.min()>-1e-4*excess[0]
    assert np.max(abs((n[:,:,0]-n[:,:,2]).sum(axis=1)-P))<2e-12


def test_actual_scheduler_recovery_converges_with_smaller_dt():
    from scipy.integrate import solve_ivp
    m=IdealBinModel(IdealBinParams(n_bins=81,zq_width_R=.15))
    _prepare_RF_only(m,.6);n0=m.n.copy();P0=m.polarizations()['P']
    def rhs(t,y):
        m.n=y.reshape(n0.shape);return m.derivative(rf_on=False,dnp_on=False).ravel()
    ref=solve_ivp(rhs,(0,4),n0.ravel(),rtol=2e-10,atol=2e-14,method='DOP853').y[:,-1].reshape(n0.shape)
    errors=[]
    for dt in [.01,.005]:
        m.n=n0.copy();m.t=0;m.params.dt=dt;m.step(round(4/dt),rf_on=False,dnp_on=False)
        errors.append(np.max(abs(m.n-ref)))
        assert abs(m.polarizations()['P']-P0)<1e-13
    assert errors[1]<.6*errors[0]


@pytest.mark.parametrize('P,Q',[(.45,.3),(-.58,.1),(0,-.25)])
def test_nonboltzmann_state_actually_relaxes_to_unique_remaining_area_Boltzmann(P,Q):
    si=pytest.importorskip('scipy.integrate')
    m=Spin1Model(Spin1Params(n_bins=81,p0=P,q0=Q,zq_width_R=.12))
    n0=m.n.copy();target=m.equilibrium_reference(P)
    def f(t,y):
        m.n=y.reshape(n0.shape)
        return m.derivative(rf_on=False,dnp_on=False).ravel()
    sol=si.solve_ivp(f,(0,3500),n0.ravel(),t_eval=[0,3500],method='DOP853',rtol=2e-10,atol=1e-14)
    assert sol.success
    m.n=sol.y[:,-1].reshape(n0.shape)
    assert abs(m.polarizations()['P']-P)<2e-11
    assert m.recovery_equilibrium_diagnostics()['fraction_rms_error']<1e-8
    assert np.max(abs((m.n-target)/m.mu[:,None]))<3e-7


def test_sparse_and_numpy_evaluators_agree_on_identical_pair_currents():
    m=Spin1Model(Spin1Params(n_bins=71))
    rng=np.random.default_rng(91)
    p=rng.random(m.n.shape);p/=p.sum(axis=1,keepdims=True);m.n=m.mu[:,None]*p
    fast=m._spin_diffusion_terms(False)
    saved=m._same_matrix;m._same_matrix=None
    try:slow=m._spin_diffusion_terms(False)
    finally:m._same_matrix=saved
    for name in fast:
        np.testing.assert_allclose(fast[name],slow[name],rtol=5e-13,atol=3e-17)


def test_Lorentzian_tail_is_not_cut_off_by_default():
    m=Spin1Model()
    assert m.params.kernel_cutoff_widths==0
    w=m.params.zq_width_R
    np.testing.assert_allclose(m._spectral_overlap(np.array([0.,w,10*w])),[1.,.5,1/101])


def test_DNP_off_on_microwave_factor_multiplies_all_internal_channels():
    m=Spin1Model(Spin1Params(n_bins=41,microwave_diffusion_factor=2.5))
    _prepare_RF_only(m,.6)
    off=m._spin_diffusion_terms(False);on=m._spin_diffusion_terms(True)
    for name in off:np.testing.assert_allclose(on[name],2.5*off[name],atol=1e-17)


def test_DNP_on_RF_off_reaches_same_unchanged_saturation_target():
    si=pytest.importorskip('scipy.integrate')
    m=Spin1Model(Spin1Params(n_bins=61,p0=.1,p_dnp_sat=.58,dnp_rate=.3,dnp_enabled=True))
    n0=m.n.copy()
    def f(t,y):
        m.n=y.reshape(n0.shape);return m.derivative(rf_on=False,dnp_on=True).ravel()
    sol=si.solve_ivp(f,(0,1500),n0.ravel(),t_eval=[1500],rtol=2e-10,atol=1e-14,method='DOP853')
    assert sol.success;m.n=sol.y[:,-1].reshape(n0.shape)
    assert abs(m.polarizations()['P']-.58)<3e-10
    assert m.recovery_equilibrium_diagnostics()['fraction_rms_error']<1e-8
