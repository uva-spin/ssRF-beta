"""Physical normalization, migration, optimizer gradients and data-selection contracts."""
from dataclasses import asdict
from pathlib import Path
import importlib.util,json
import numpy as np
import pytest
from ssrf_realtime.multisite_lineshape import powder_branch,site_components
from ssrf_realtime.material_config import MaterialConfig,PopulationSnapshot,GROUPS,_state_digest
from ssrf_realtime.ideal_model import IdealBinModel,IdealBinParams
from ssrf_realtime.tensor_optimizer import FrozenDynamics

ROOT=Path(__file__).resolve().parents[1]


def material():return MaterialConfig.load(ROOT/'tests/fixtures/materials/target_data_butanol.json')


@pytest.mark.parametrize('epsilon',[-1,1])
@pytest.mark.parametrize('eta',[0.,.065,.32])
def test_powder_density_matches_pinned_upstream_reference(epsilon,eta):
    file=ROOT/'tests/fixtures/nmr_ai_complex_lineshape_reference.py'
    spec=importlib.util.spec_from_file_location('upstream_powder',file)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    x=np.r_[np.linspace(-7,7,100),[-1.07,-1.,-.93,.93,1.,1.07]]
    np.testing.assert_allclose(powder_branch(x,epsilon,eta,.02,128),
        ref.powder_complex(x,epsilon,eta,.02,128),rtol=2e-14,atol=1e-15)


def test_multisite_weights_integrated_area_not_peak_height():
    from scipy.integrate import quad
    for site,expected in ((0,.8),(1,.2)):
        def density(r):return -site_components(r,cd_scale=.8,od_ratio=1.7,
            hwhm=.07,eta_cd=.065,eta_od=.12,od_fraction=.2)[site].imag
        area=quad(density,-np.inf,np.inf,epsabs=2e-8,limit=300)[0]
        assert area==pytest.approx(expected,abs=2e-7)


def test_multisite_mass_signed_spectra_and_boltzmann_stationarity():
    c=material();m=c.make_model(preserve_state=False)
    assert m.n.shape==(500,3);assert m.mu.sum()==pytest.approx(1.)
    assert m.mu.min()>0
    np.testing.assert_allclose(m.n.sum(axis=1),m.mu,atol=1e-17)
    for term in m._spin_diffusion_terms(False).values():assert abs(term).max()<1e-17
    c.parameters['p0']=-m.params.p0;negative=c.make_model(preserve_state=False)
    np.testing.assert_allclose(negative.spectrum()[3],-m.spectrum()[3][::-1],rtol=2e-12,atol=1e-15)
    np.testing.assert_allclose(m.static_plot_signal_reference()[3],m.reference_spectrum()[3])


def test_multisite_gradient_and_live_rhs_on_manipulated_populations():
    m=material().make_model(preserve_state=False);rng=np.random.default_rng(5127)
    m.n=m.mu[:,None]*rng.dirichlet([3,2,1],500);m.params.dnp_enabled=True
    dyn=FrozenDynamics(PopulationSnapshot.capture(m));u=rng.random(500)/500
    m._command_override=u
    np.testing.assert_allclose(dyn.rhs(m.n,u),m.derivative(rf_on=True),rtol=2e-11,atol=1e-15)
    adj=rng.normal(size=m.n.shape);dn=rng.normal(size=m.n.shape)*m.mu[:,None];du=rng.normal(size=500)/500
    gn,gu=dyn.vjp(m.n,u,adj);eps=1e-6
    fd=np.sum(adj*(dyn.rhs(m.n+eps*dn,u+eps*du)-dyn.rhs(m.n-eps*dn,u-eps*du)))/(2*eps)
    assert fd==pytest.approx(np.sum(gn*dn)+gu@du,rel=2e-7,abs=1e-10)


def test_old_snapshot_migration_authenticates_before_adding_multisite_defaults(tmp_path):
    m=IdealBinModel();s=PopulationSnapshot.capture(m)
    for key in GROUPS['multisite_lineshape']:s.parameters.pop(key)
    s.fingerprint=_state_digest(s.parameters,s.n,s.mu);s.save(tmp_path/'old.npz')
    migrated=PopulationSnapshot.load(tmp_path/'old.npz');assert migrated.matches(m)
    np.testing.assert_array_equal(migrated.make_model().n,m.n)
    s.parameters['diffusion_scale']*=2;s.save(tmp_path/'tampered.npz')
    with pytest.raises(ValueError,match='fingerprint'):PopulationSnapshot.load(tmp_path/'tampered.npz')


@pytest.mark.parametrize('key,value',[('line_model','multisite_butanol'),('line_od_fraction',.1),
    ('line_eta_cd',.1),('line_cd_scale_R',1.2),('line_phi_points',256)])
def test_capacity_changes_require_explicit_reinitialization(key,value):
    m=IdealBinModel();c=MaterialConfig.capture(m);c.parameters[key]=value
    with pytest.raises(ValueError,match='capacities'):c.make_model(m,preserve_state=True)


def test_unsupported_site_geometry_and_partial_schema_fail():
    c=material();c.parameters['orientation_corr_fraction']=.2
    with pytest.raises(ValueError,match='site-resolved'):c.validate()
    data=material().to_dict();data['parameters']['multisite_lineshape'].pop('line_eta_od')
    with pytest.raises(ValueError,match='Keys in'):MaterialConfig.from_dict(data)


def test_multisite_reference_program_uses_new_grid_and_preserved_budget():
    from ssrf_realtime.pulse_program import PulseProgram
    c=material();m=c.make_model(preserve_state=False)
    program=PulseProgram.load(ROOT/'tests/fixtures/programs/target_data_v5_reference.json')
    program.validate_grid(m.Rplus)
    assert program.compile().envelope.sum()==pytest.approx(1.)
    assert program.compile().exposure.sum()==pytest.approx(34.)
    old=MaterialConfig.load(ROOT/'tests/fixtures/materials/experimental_event_1406922415.json')
    assert c.optimizer==old.optimizer
    for key in GROUPS['recovery']:
        # The fixed K is carried through log/exp by the calibration fitter.
        # Allow its one-ULP round trip without allowing a material retune.
        if isinstance(c.parameters[key],float):
            assert c.parameters[key]==pytest.approx(old.parameters[key],rel=2e-15,abs=0.)
        else:assert c.parameters[key]==old.parameters[key]
