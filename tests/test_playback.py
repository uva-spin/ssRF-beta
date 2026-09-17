"""Regression for the reported load -> Run simulation -> RF ON no-op.

These exercise the real, unmodified scheduler and population model through the
new Qt-independent playback controller. GUI callback tests are in the next file.
"""
from pathlib import Path
import numpy as np
import pytest
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.pulse_program import BinPulse, RFProfile, PulseProgram, example_program
from ssrf_realtime.playback import PulsePlaybackController

ROOT = Path(__file__).resolve().parents[1]


def make_model(duration=.03, start=0., rate=2., dt=.003):
    m = IdealBinModel(IdealBinParams(n_bins=41,dt=dt,diffusion_enabled=False,
                                   dnp_enabled=False,t1_rate=0.))
    m.set_program(PulseProgram(41,-3.,3.,profiles=[RFProfile('Test',pulses=[
        BinPulse(14,rate,start,duration),BinPulse(27,rate*.3,start,duration)])]))
    return m


def test_reproduce_original_gate_only_noop_then_fixed_start():
    m = make_model()
    m.step(30)  # Running the simulation before RF must not consume a READY program.
    before = m.n.copy()
    m.set_rf_enabled(True)  # Old RF-button implementation.
    m.step(5)
    assert m.program_state == 'ready'
    assert np.count_nonzero(m.applied_rf_field()) == 0
    assert not np.any(m.delivered_exposure)
    np.testing.assert_allclose(m.n,before,rtol=0,atol=1e-17)
    t = m.t
    assert PulsePlaybackController(m).rf_on() == 'started'
    assert m._epoch == t
    PulsePlaybackController(m).advance(2)
    assert m.delivered_exposure.sum() > 0
    assert np.max(abs(m.n-before)) > 1e-6


@pytest.mark.parametrize('source',['generic','horn_pedestal','generated_json','generated_csv'])
def test_all_program_sources_start_from_rf_button(source):
    m = IdealBinModel(IdealBinParams(diffusion_enabled=False,dt=.0015))
    if source == 'horn_pedestal':
        m.set_program(PulseProgram.load(ROOT/'examples/horn_and_pedestal.json'))
    elif source.startswith('generated'):
        path = ROOT/'outputs/default_701_design'/('rf_program.json' if source.endswith('json') else 'rf_program.csv')
        m.set_program(PulseProgram.load(path) if source.endswith('json') else PulseProgram.load_csv(path))
    before = m.n.copy()
    m.step(10)
    control = PulsePlaybackController(m)
    control.rf_on()
    # First pulse may have a deliberate leading delay.
    first = min(p.start for pr in m.program.profiles if pr.enabled
                for p in pr.pulses if p.enabled and p.rate>0 and p.duration>0)
    stop = min(m._compiled.end_time,first+.02)
    while m.program_elapsed < stop:
        control.advance(1,pause_at_end=True)
    assert m.delivered_exposure.sum() > 0
    assert np.max(abs(m.n-before)) > 1e-7


def test_load_and_run_without_rf_does_not_consume_program():
    m = make_model()
    control = PulsePlaybackController(m)
    before=m.n.copy()
    control.advance(100)
    assert m.program_state == 'ready'
    assert m.program_elapsed==0
    assert not m.params.rf_enabled
    assert not np.any(m.delivered_exposure)
    np.testing.assert_allclose(m.n,before,atol=2e-17,rtol=0)


def test_start_does_not_change_population_or_gain():
    m=make_model()
    m.program.gain=1.7
    n=m.n.copy();reference=m.reference_n.copy() if hasattr(m,'reference_n') else None
    PulsePlaybackController(m).rf_on()
    np.testing.assert_array_equal(m.n,n)
    assert m.program.gain==1.7


def test_finished_program_disarms_and_rf_on_replays_once():
    m=make_model()
    c=PulsePlaybackController(m);c.rf_on()
    update=c.advance(30)
    assert update.completed and not update.pause_requested
    assert m.program_state=='finished' and not m.params.rf_enabled
    old_total=m.total_delivered_exposure.copy();n=m.n.copy();t=m.t
    assert c.rf_on()=='restarted'
    assert m._epoch==t
    assert not np.any(m.delivered_exposure)
    np.testing.assert_array_equal(n,m.n)
    c.advance(30)
    np.testing.assert_allclose(m.total_delivered_exposure,2*old_total,rtol=1e-13,atol=1e-14)


def test_muting_then_resuming_does_not_restart_running_schedule():
    m=make_model(duration=.3)
    c=PulsePlaybackController(m);c.rf_on();c.advance(5)
    epoch=m._epoch;exposure=m.delivered_exposure.copy()
    c.rf_off();c.advance(5)
    np.testing.assert_array_equal(exposure,m.delivered_exposure)
    elapsed=m.program_elapsed
    assert c.rf_on()=='resumed'
    assert m._epoch==epoch and m.program_elapsed==elapsed
    c.advance(5)
    np.testing.assert_allclose(m.delivered_exposure,2*exposure,rtol=1e-13,atol=1e-14)


def test_program_finishing_while_muted_can_be_replayed():
    m=make_model();c=PulsePlaybackController(m)
    c.rf_on();c.advance(2);c.rf_off();update=c.advance(30)
    assert update.completed and m.program_state=='finished'
    assert not m.params.rf_enabled
    c.rf_on();c.advance(30)
    assert m.delivered_exposure.sum() > 0


def test_waiting_period_has_no_accidental_restart():
    m=make_model(start=.03,duration=.03)
    c=PulsePlaybackController(m);c.rf_on()
    epoch=m._epoch
    assert m.program_state=='waiting' and 'WAITING' in c.status_text()
    c.advance(2)
    assert not np.any(m.delivered_exposure)
    c.rf_off();assert c.rf_on()=='resumed'
    assert m._epoch==epoch
    while m.program_state!='finished':c.advance(1,pause_at_end=True)
    expected=m.program.expected_exposure() if hasattr(m.program,'expected_exposure') else None
    assert m.delivered_exposure.sum()==pytest.approx(.03*(2.+.6))


def test_gain_zero_rejected_without_rf_on():
    m=make_model();m.set_program_gain(0.)
    with pytest.raises(ValueError,match='gain is zero'):
        PulsePlaybackController(m).rf_on()
    assert not m.params.rf_enabled and m.program_state=='ready'


@pytest.mark.parametrize('rate,duration',[(0.,.1),(2.,0.)])
def test_empty_or_disabled_commands_rejected(rate,duration):
    m=make_model(rate=rate,duration=duration)
    with pytest.raises(ValueError,match='No enabled pulses'):
        PulsePlaybackController(m).rf_on()
    assert not m.params.rf_enabled


def test_hold_endpoint_does_not_advance_recovery_or_change_dt():
    m=make_model(duration=.0017,dt=.003)
    before=m.n.copy();dt=m.params.dt
    c=PulsePlaybackController(m);c.rf_on();end=m._event_times_absolute[-1]
    update=c.advance(100,pause_at_end=True)
    assert update.completed and update.pause_requested
    assert m.t==end and m.params.dt==dt
    assert m.program_state=='finished' and not m.params.rf_enabled
    assert m.delivered_exposure.sum()==pytest.approx(.0017*(2.+.6))
    assert np.max(abs(m.n-before))>0
    # Recovery playback is a separate deliberate Run action.
    c.advance(3,pause_at_end=True)
    assert m.t > end and not m.params.rf_enabled


def test_unheld_completion_has_identical_dynamics_to_previous_scheduler():
    a=make_model(duration=.03);b=make_model(duration=.03)
    a.params.diffusion_enabled=True;b.params.diffusion_enabled=True
    c=PulsePlaybackController(a);c.rf_on();b.start_program()
    for _ in range(5):
        c.advance(12);b.step(12)
        np.testing.assert_array_equal(a.n,b.n)
    np.testing.assert_array_equal(a.delivered_exposure,b.delivered_exposure)
    assert not a.params.rf_enabled


def test_status_exposes_ready_waiting_muted_finished_and_receipt():
    m=make_model(start=.006);c=PulsePlaybackController(m)
    assert 'READY' in c.status_text() and 'click to start' in c.button_text()
    c.rf_on();assert 'WAITING' in c.status_text()
    c.advance(3);assert 'RUNNING' in c.status_text()
    c.rf_off();assert 'MUTED' in c.status_text()
    c.advance(100);assert 'FINISHED' in c.status_text()
    assert 'Delivered this run: 2 bins' in c.status_text()
    assert 'replay' in c.button_text()
