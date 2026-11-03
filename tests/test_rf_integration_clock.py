"""Fine RF integration must not leave the simulation/recovery clock slowed."""
import copy
import numpy as np
import pytest
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot
from ssrf_realtime.playback import PulsePlaybackController
from ssrf_realtime.pulse_program import PulseProgram, RFProfile, BinPulse
from ssrf_realtime.tensor_optimizer import replay


def setup_model(start=0.,duration=.01):
    model=IdealBinModel(IdealBinParams(n_bins=61,dt=.0015,
        rf_response_mode='spectral',rf_peak_selectivity=.7))
    program=PulseProgram(61,-3.,3.,profiles=[RFProfile('Fine RF',pulses=[
        BinPulse(20,20.,start,duration),BinPulse(43,10.,start,duration)])])
    model.set_program(program,integration_dt=model.params.dt/64)
    return model


def test_64_times_finer_RF_matches_reference_and_keeps_normal_DNP_recovery_clock():
    model=setup_model();normal=model.params.dt;fine=model.program_integration_dt
    snapshot=PopulationSnapshot.capture(model);controller=PulsePlaybackController(model)
    controller.rf_on();update=controller.advance(12,pause_at_end=True)
    assert update.completed and update.pause_requested
    assert model.t==pytest.approx(.01)
    assert model.params.dt==normal
    reference,_=replay(snapshot,model.program,.01,dt=fine)
    np.testing.assert_allclose(model.n,reference,atol=1e-14,rtol=2e-12)
    assert model.delivered_exposure.sum()==pytest.approx(.3)

    # Both models now start at the same burnt populations and DNP-on state.
    # The RF-only integration ceiling must have no effect on recovery.
    model.params.dnp_enabled=True
    control=copy.deepcopy(model);control.program_integration_dt=None
    t=model.t;P=np.sum(model.n[:,0]-model.n[:,2])
    controller.advance(12)
    control.step(12)
    assert model.t-t==pytest.approx(12*normal)
    assert model.last_substeps==12
    np.testing.assert_array_equal(model.n,control.n)
    assert np.sum(model.n[:,0]-model.n[:,2])>P
    assert MaterialConfig.capture(model).parameters['dt']==normal


@pytest.mark.parametrize('phase',['ready','waiting','muted','stopped'])
def test_fine_RF_limit_is_inactive_when_no_RF_is_delivered(phase):
    m=setup_model(start=.1 if phase=='waiting' else 0.,duration=.5)
    c=PulsePlaybackController(m)
    if phase in ('waiting','muted','stopped'):c.rf_on()
    if phase=='muted':c.rf_off()
    if phase=='stopped':m.stop_program()
    c.advance(12)
    assert m.t==pytest.approx(12*.0015)
    assert m.last_substeps==12
    assert not m.delivered_exposure.any()


def test_replay_reset_and_material_apply_preserve_program_limit_but_replacement_clears_it():
    m=setup_model();fine=m.program_integration_dt
    c=PulsePlaybackController(m);c.rf_on();c.advance(12,pause_at_end=True)
    c.rf_on();c.advance(1)
    assert m.last_substeps>=63
    assert m.program_integration_dt==fine
    m.reset();assert m.program_integration_dt==fine
    config=MaterialConfig.capture(m);restored=config.make_model(m,preserve_state=True)
    assert restored.program_integration_dt==fine
    assert restored.params.dt==.0015
    m.set_program(m.program)
    assert m.program_integration_dt is None


@pytest.mark.parametrize('step',[0,-1,float('nan'),float('inf'),True])
def test_invalid_RF_limit_is_rejected_transactionally(step):
    m=setup_model();program=m.program;fine=m.program_integration_dt
    with pytest.raises(ValueError,match='integration_dt'):
        m.set_program(program,integration_dt=step)
    assert m.program is program and m.program_integration_dt==fine
