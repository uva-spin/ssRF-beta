"""RF profile regularity is subordinate to actual population gain and limits."""
from dataclasses import asdict
import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.integrate import solve_ivp

from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.material_config import PopulationSnapshot, MaterialConfig
from ssrf_realtime.pulse_program import PulseProgram
from ssrf_realtime.tensor_optimizer import (OptimizerSettings, optimize_tensor, replay,
    polarizations, smooth_bounded_commands, first_order_gap)
from ssrf_realtime.tensor_optimizer import FrozenDynamics
from fixture_materials import default_material


def test_bounded_smoothing_preserves_power_and_forbidden_bins():
    rng=np.random.default_rng(24)
    upper=rng.uniform(.01,.2,71);upper[10:20]=0;upper[45:50]=0
    u=rng.random(71)*upper
    for sigma in (.4,2.,10.,1e6):
        candidate=smooth_bounded_commands(u,upper,sigma)
        assert np.all(candidate>=0) and np.all(candidate<=upper)
        assert np.all(candidate[upper==0]==0)
        assert candidate.sum()==pytest.approx(u.sum(),abs=1e-13)
    np.testing.assert_array_equal(smooth_bounded_commands(u*0,upper,2.),u*0)


@pytest.mark.parametrize('cap',[None,0.,.5,4.])
def test_first_order_gap_matches_independent_linear_program(cap):
    rng=np.random.default_rng(73);upper=rng.uniform(.01,.3,31);g=rng.normal(size=31)
    u=.4*upper
    if cap is not None and u.sum()>cap:u*=cap/u.sum()
    optimum=linprog(-g,A_ub=None if cap is None else np.ones((1,31)),
        b_ub=None if cap is None else [cap],bounds=list(zip(np.zeros(31),upper)),method='highs')
    assert optimum.success
    assert first_order_gap(u,g,upper,cap)==pytest.approx(-optimum.fun-g@u,abs=1e-12)
    assert first_order_gap(optimum.x,g,upper,cap)<1e-12


def test_old_materials_disable_refinement_and_invalid_width_is_rejected():
    old=asdict(OptimizerSettings());del old['profile_smoothing_fwhm_R']
    assert OptimizerSettings.from_dict(old).profile_smoothing_fwhm_R==0
    for value in (-1,float('nan'),True):
        with pytest.raises(ValueError,match='profile_smoothing'):
            OptimizerSettings(profile_smoothing_fwhm_R=value).validate()


@pytest.mark.parametrize('p0',[.45,-.45])
def test_refined_profile_is_real_replayable_RF_with_a_checked_Q_loss(tmp_path,p0):
    m=IdealBinModel(IdealBinParams(n_bins=81,p0=p0,rf_response_mode='spectral',
        rf_homogeneous_hwhm_R=.08,diffusion_enabled=False,dt=.003))
    s=PopulationSnapshot.capture(m)
    settings=OptimizerSettings(max_power=2,max_total_power=1,max_total_exposure=.4,
        min_duration=.4,max_duration=.4,duration_samples=2,duration_refinements=0,
        starts=1,max_iterations=60,search_dt=.003,profile_smoothing_fwhm_R=.12)
    result=optimize_tensor(s,settings)
    refinement=result.report['profile_refinement']
    assert refinement['accepted']
    assert refinement['roughness_reduction_fraction']>0
    assert refinement['Q_loss']<=refinement['allowed_Q_loss']
    assert result.report['final']['Q']>=refinement['minimum_Q_required']
    assert refinement['best_duration_reference_Q']-result.report['final']['Q']<=refinement['global_Q_allowance']
    assert result.powers.sum()<=1*(1+1e-12)
    assert result.duration*result.powers.sum()<=.4*(1+1e-12)
    np.testing.assert_array_equal(result.powers[~result.candidates],0)
    result.export(tmp_path)
    raw=PulseProgram.load(tmp_path/'unrefined_rf_program.json')
    final=PulseProgram.load(tmp_path/'rf_program.json')
    n_raw,_=replay(s,raw,result.duration,dt=result.report['recommended_playback_dt'])
    n_final,_=replay(s,final,result.duration,dt=result.report['recommended_playback_dt'])
    np.testing.assert_allclose(n_final,result.final_n,atol=1e-16)
    assert polarizations(n_raw)['Q']-polarizations(n_final)['Q']==pytest.approx(refinement['Q_loss'],abs=1e-13)
    assert MaterialConfig.load(tmp_path/'material.json').optimizer['profile_smoothing_fwhm_R']==.12
    np.testing.assert_array_equal(m.n,s.n)


def test_hard_single_carrier_constraint_is_not_blurred_away():
    m=IdealBinModel(IdealBinParams(n_bins=61,diffusion_enabled=False))
    j=int(np.argmin(abs(m.Rplus+.9)));r=float(m.Rplus[j])
    result=optimize_tensor(PopulationSnapshot.capture(m),OptimizerSettings(
        min_duration=.1,max_duration=.1,duration_samples=2,duration_refinements=0,
        regions=[[r,r]],max_iterations=40,starts=1,profile_smoothing_fwhm_R=.2))
    assert not result.report['profile_refinement']['accepted']
    assert np.count_nonzero(result.powers)==1 and result.powers[j]>0


def test_timeout_keeps_the_feasible_candidate_in_the_interrupted_horizon(monkeypatch):
    import ssrf_realtime.tensor_optimizer as module
    def interrupted(objective,x0,**kwargs):
        objective(x0)
        raise TimeoutError('Synthetic deadline after a valid endpoint evaluation')
    monkeypatch.setattr(module,'minimize',interrupted)
    m=IdealBinModel(IdealBinParams(n_bins=41,diffusion_enabled=False))
    result=optimize_tensor(PopulationSnapshot.capture(m),OptimizerSettings(
        min_duration=.1,max_duration=.2,duration_samples=2,duration_refinements=0,starts=1))
    assert result.report['budget_reached']
    assert result.duration==.1 and result.powers.any()
    assert result.report['delta_Q']>0
    assert 'retained' in result.scan[0]['message']


def test_short_strong_pulse_playback_accuracy_against_independent_solver():
    # The old half-step criterion accepted a full-step error almost twice the
    # requested tolerance for this normal 45% demonstration starting state.
    material=default_material()
    snapshot=PopulationSnapshot.capture(material.make_model(preserve_state=False))
    settings=OptimizerSettings.from_dict(material.optimizer)
    settings.min_duration=settings.max_duration=.01
    settings.duration_refinements=0;settings.starts=1;settings.max_iterations=30
    result=optimize_tensor(snapshot,settings)
    assert result.report['status']=='completed'
    dyn=FrozenDynamics(snapshot)
    sol=solve_ivp(lambda t,y:dyn.rhs(y.reshape(dyn.N,3),result.powers).ravel(),
        (0,result.duration),snapshot.n.ravel(),method='DOP853',rtol=2e-10,
        atol=np.repeat(snapshot.mu*1e-12,3))
    assert sol.success
    independent_Q=polarizations(sol.y[:,-1].reshape(dyn.N,3))['Q']
    assert abs(independent_Q-result.report['final']['Q'])<settings.convergence_tolerance
    assert result.report['time_step_check']['estimated_playback_Q_error']<=settings.convergence_tolerance
