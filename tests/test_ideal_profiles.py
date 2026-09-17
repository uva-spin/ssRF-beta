"""Independent checks of the RF input and event timing, not an imposed gain rule."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm

from ssrf_realtime.model import Spin1Model
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.pulse_program import BinPulse, RFProfile, PulseProgram, make_profile, example_program


def setup_model(pulses=None, **kwargs):
    params=IdealBinParams(n_bins=101, dt=0.0015, diffusion_enabled=False, **kwargs)
    program=PulseProgram(params.n_bins,params.r_min,params.r_max,
                         profiles=[RFProfile('test',pulses=pulses or [])])
    return IdealBinModel(params,program)


def test_profile_shapes_and_exact_support():
    grid=np.linspace(-3,3,101)
    for shape in ('flat','linear','triangle','gaussian'):
        profile=make_profile(grid,'region',-1,-0.5,shape,2,4,start=0.2,duration=0.3,start_step=0.02,duration_right=0.8)
        lo,hi=(int(np.argmin(abs(grid-r))) for r in (-1,-0.5))
        assert [p.bin_index for p in profile.pulses]==list(range(lo,hi+1))
        assert all(0<=p.rate<=4 for p in profile.pulses)
        assert profile.pulses[0].start==0.2
        assert profile.pulses[-1].start==pytest.approx(0.2+(hi-lo)*0.02)
        assert profile.pulses[-1].duration==pytest.approx(0.8)
    assert max(p.rate for p in make_profile(grid,'t',-1,0,'triangle',3).pulses)==3


def test_compiler_additive_overlap_repeated_pulses_and_half_open_edges():
    program=PulseProgram(101,-3,3, profiles=[RFProfile('a',pulses=[BinPulse(10,2,0.2,0.3),BinPulse(10,4,0.7,0.1)]),
                                           RFProfile('b',pulses=[BinPulse(10,3,0.3,0.4),BinPulse(20,7,0.3,0.4)])])
    c=program.compile()
    assert c.field_at(0.199)[10]==0
    assert c.field_at(0.2)[10]==2
    assert c.field_at(0.3)[10]==5
    assert c.field_at(0.5)[10]==3
    assert c.field_at(0.7)[10]==4
    assert np.all(c.field_at(1.0)==0)
    assert c.envelope[10]==5
    assert c.exposure[10]==pytest.approx(2*0.3+3*0.4+4*0.1)
    assert c.field_at(0.35)[20]==7   # can evaluate out of order


def test_disabled_profiles_rows_zero_durations_and_gain():
    prog=PulseProgram(101,-3,3, gain=2, profiles=[
        RFProfile('disabled',False,[BinPulse(1,9,0,1)]),
        RFProfile('on',True,[BinPulse(2,4,0,1,False),BinPulse(3,5,0,0),BinPulse(4,2,0,0.5)])])
    m=IdealBinModel(IdealBinParams(n_bins=101),prog)
    m.start_program()
    u=m.commanded_rf_field()
    assert u[4]==4 and np.count_nonzero(u)==1


def test_json_roundtrip_and_grid_rejection(tmp_path):
    p=example_program(np.linspace(-3,3,101))
    p.profiles.append(RFProfile('empty'))
    p.gain=1.3
    path=tmp_path/'p.json';p.save(path)
    q=PulseProgram.load(path)
    assert p.to_dict()==q.to_dict()
    with pytest.raises(ValueError,match='grid'):
        q.validate_grid(np.linspace(-3,3,99))
    bad=p.to_dict();bad['profiles'][0]['pulses'][0]['rate']=float('nan')
    with pytest.raises(ValueError):PulseProgram.from_dict(bad)
    bad=p.to_dict();bad['profiles'][0]['enabled']='False'
    with pytest.raises(ValueError):PulseProgram.from_dict(bad)


def test_csv_roundtrip_preserves_pulses_gain_and_enable(tmp_path):
    p=example_program(np.linspace(-3,3,101));p.gain=0.7
    p.profiles[0].pulses[0].enabled=False
    path=tmp_path/'p.csv';p.save_csv(path)
    q=PulseProgram.load_csv(path)
    assert p.to_dict()==q.to_dict()


@pytest.mark.parametrize('pulse',[BinPulse(-1,2,0,1),BinPulse(200,2,0,1),BinPulse(2,-1,0,1),
                                  BinPulse(2,2,-1,1),BinPulse(2,2,0,-1),BinPulse(2,2,float('inf'),1),
                                  BinPulse(2.5,2,0,1),BinPulse(2,2,0,1,'false')])
def test_bad_pulses_rejected(pulse):
    with pytest.raises(ValueError):pulse.validate(101)


def test_rf_operator_matches_three_level_matrix_at_multiple_bins():
    # Address both signs as well as R=0; check all three levels directly.
    m=setup_model([BinPulse(35,1.3,0,1),BinPulse(65,0.7,0,1),BinPulse(50,2.1,0,1)])
    m.start_program()
    gp,gm=m.rf_rate_fields()
    actual=m._rf_population_term(True)
    expected=np.zeros_like(actual)
    for k in range(len(m.n)):
        A=np.array([[-gp[k],gp[k],0],[gp[k],-gp[k]-gm[k],gm[k]],[0,gm[k],-gm[k]]])
        expected[k]=A@m.n[k]
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-16)
    np.testing.assert_allclose(actual.sum(axis=1),0,atol=1e-16)
    assert np.count_nonzero(np.any(actual!=0,axis=1))==3
    np.testing.assert_array_equal(m.commanded_rf_field()[[34,36,64,66]],0)


def test_same_physical_command_for_both_branches():
    m=setup_model([BinPulse(35,2.7,0,1)],capacity_rate_power=0)
    m.start_program()
    gp,gm=m.rf_rate_fields()
    assert gp[35]==gm[65]==2.7
    assert gp[65]==gm[35]==0


@pytest.mark.parametrize('dnp_on,diff_on',[(False,False),(True,False),(False,True),(True,True)])
def test_single_bin_regression_with_working_model(dnp_on,diff_on):
    p=IdealBinParams(n_bins=101, dt=0.0015, rf_burn_R=0.42,
                     dnp_enabled=dnp_on,diffusion_enabled=diff_on)
    baseline=Spin1Model(replace(p))
    ideal=IdealBinModel(replace(p),PulseProgram(101,-3,3,profiles=[RFProfile('a',pulses=[BinPulse(57,p.gamma_rf,0,2)])]))
    baseline.set_rf_enabled(True);ideal.start_program()
    for _ in range(200):
        baseline.step();ideal.step()
    np.testing.assert_allclose(ideal.n,baseline.n,rtol=0,atol=2e-16)
    assert ideal.t==baseline.t


def test_dnp_and_relaxation_untouched_when_program_ready():
    p=IdealBinParams(n_bins=101,dnp_enabled=True,t1_rate=0.12,t1_p_eq=-0.2)
    baseline=Spin1Model(replace(p));ideal=IdealBinModel(replace(p))
    for _ in range(120):baseline.step();ideal.step()
    np.testing.assert_allclose(ideal.n,baseline.n,rtol=0,atol=2e-16)


def test_sub_dt_pulse_receives_full_duration_without_timing_roundoff():
    m=setup_model([BinPulse(32,7.0,0.00317,0.000019),BinPulse(42,3.0,0.00413,0.00278)])
    m.params.dt=0.01
    m.start_program()
    m.step(2)
    assert m.program_state=='finished'
    assert m.delivered_exposure[32]==pytest.approx(7*0.000019,abs=2e-16)
    assert m.delivered_exposure[42]==pytest.approx(3*0.00278,abs=2e-16)
    assert np.count_nonzero(m.delivered_exposure)==2
    assert np.all(m.applied_rf_field()==0)


def test_fractional_pulse_timing_after_nonzero_epoch():
    m=setup_model([BinPulse(35,2.0,0.0013,0.0057)])
    m.step(7)
    t=m.t;n=m.n.copy();m.start_program()
    assert m.t==t
    np.testing.assert_array_equal(m.n,n)
    m.step(12)
    assert m.delivered_exposure[35]==pytest.approx(2*0.0057,abs=2e-16)


def test_rf_only_endpoint_converges_to_matrix_exponential():
    m=setup_model([BinPulse(35,2.0,0,0.2),BinPulse(65,0.5,0,0.2)],capacity_rate_power=0)
    m.params.dt=0.0002
    initial=m.n.copy();m.start_program();m.step(1001)
    for k,gp,gm in [(35,2,0.5),(65,0.5,2)]:
        A=np.array([[-gp,gp,0],[gp,-gp-gm,gm],[0,gm,-gm]])
        np.testing.assert_allclose(m.n[k],expm(A*0.2)@initial[k],rtol=2e-4,atol=1e-8)
    np.testing.assert_allclose(m.n.sum(axis=1),m.mu,atol=1e-16)


def test_rf_off_mutes_but_program_and_recovery_continue():
    m=setup_model([BinPulse(35,2,0,0.06)])
    m.start_program();m.step(10);x=m.delivered_exposure.copy()
    m.set_rf_enabled(False);before=m.t;m.step(50)
    assert m.t>before and m.program_state=='finished'
    np.testing.assert_array_equal(m.delivered_exposure,x)
    assert np.all(m.applied_rf_field()==0)
    m.start_program();m.step(5)
    assert m.delivered_exposure[35]==pytest.approx(2*5*m.params.dt)


def test_monitor_selection_does_not_move_program_or_clear_populations():
    m=setup_model([BinPulse(35,2,0,1)])
    m.start_program();m.step(10)
    field=m.applied_rf_field();state=m.n.copy();epoch=m._epoch
    m.params.rf_burn_R=1.4
    np.testing.assert_array_equal(m.applied_rf_field(),field)
    np.testing.assert_array_equal(m.n,state)
    assert m._epoch==epoch


def test_stop_and_install_are_not_population_resets():
    m=setup_model([BinPulse(35,2,0,0.03)])
    m.start_program();m.step(5)
    state=m.n.copy();t=m.t
    m.stop_program()
    assert not m.params.rf_enabled and m.program_state=='stopped'
    np.testing.assert_array_equal(m.n,state)
    p=m.program.clone();p.profiles[0].pulses[0].rate=3
    m.set_program(p)
    np.testing.assert_array_equal(m.n,state)
    assert m.t==t


def test_internal_diffusion_conserves_reduced_P_after_program_end():
    m=setup_model([BinPulse(35,3,0,0.1)])
    m.start_program();m.step(70)
    m.params.diffusion_enabled=True
    m.params.diffusion_scale=10
    pol=m.polarizations();state=m.n.copy()
    m.step(200)
    assert m.polarizations()['P']==pytest.approx(pol['P'],abs=5e-14)
    assert not np.array_equal(state,m.n)


def test_large_commands_trigger_subdivision_not_population_clipping():
    m=setup_model([BinPulse(35,10000,0,0.001)])
    m.params.dt=0.005
    m.start_program();m.step()
    assert m.last_substeps>5
    assert np.all(m.n>=0)
    np.testing.assert_allclose(m.n.sum(axis=1),m.mu,atol=1e-16)
    assert m.delivered_exposure[35]==pytest.approx(10,rel=1e-13)


def test_no_voigt_function_used(monkeypatch):
    import ssrf_realtime.model as module
    def fail(*a,**kw): raise AssertionError('Voigt must not be evaluated')
    monkeypatch.setattr(module,'bin_averaged_voigt',fail)
    m=setup_model([BinPulse(35,2,0,1)]);m.start_program();m.step(5)
    m.rf_profile_physical();m.rf_profile_arrays();m.effective_local_rates();m.selected_rate_balance()


def test_gain_is_explicit_no_automatic_profile_normalization():
    m=setup_model([BinPulse(10,7,0,1),BinPulse(20,1,0,1),BinPulse(10,3,0,1)])
    m.start_program()
    assert m.commanded_rf_field()[10]==10 and m.commanded_rf_field()[20]==1
    m.set_program_gain(0.2)
    assert m.commanded_rf_field()[10]==2 and m.commanded_rf_field()[20]==0.2


def test_negative_polarization_and_simultaneous_mirror_drive():
    m=setup_model([BinPulse(35,2,0,0.1),BinPulse(65,2,0,0.1)],p0=-0.45)
    before=m.polarizations()['P'];m.start_program();m.step(80)
    assert before<m.polarizations()['P']<0
    assert np.all(m.n>=0)


def test_preserved_base_sources_match_manifest():
    import hashlib
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'BASELINE_MANIFEST.json').read_text())
    for name,digest in manifest['unchanged_baseline_files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
