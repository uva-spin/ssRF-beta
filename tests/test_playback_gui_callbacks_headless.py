"""Run the ACTUAL GUI callback bodies without Qt rendering.

The callbacks are compiled from the shipped GUI source, not rewritten here.
Small widget doubles implement setChecked/toggled/blockSignals behavior so
load -> run -> RF ON can be checked even on a host without a Qt binding.
These tests do not claim to test Qt painting, mouse dispatch or native windows.
The complementary test_playback_gui_optional.py exercises actual Qt buttons.
"""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from ssrf_realtime.playback import PulsePlaybackController
from ssrf_realtime.ideal_model import IdealBinModel,IdealBinParams
from ssrf_realtime.material_config import PopulationSnapshot,MaterialConfig
from ssrf_realtime.tensor_optimizer import OptimizerSettings
from ssrf_realtime.pulse_program import PulseProgram,BinPulse,RFProfile

ROOT=Path(__file__).resolve().parents[1]


class Widget:
    def __init__(self,value=0,callback=None):
        self.checked=False;self.value_=value;self.callback=callback;self.blocked=False
        self.text='';self.tooltip='';self.dec=5;self.lo=1e-5;self.changes=0
    def blockSignals(self,flag):
        old=self.blocked;self.blocked=flag;return old
    def isChecked(self):return self.checked
    def setChecked(self,state):
        changed=bool(state)!=self.checked;self.checked=bool(state)
        if changed and not self.blocked and self.callback:
            self.changes+=1;self.callback(self.checked)
    def click(self):self.setChecked(not self.checked)
    def setText(self,text):self.text=text
    def setStyleSheet(self,text):pass
    def setToolTip(self,text):self.tooltip=text
    def setValue(self,value):self.value_=value
    def setMinimum(self,x):self.lo=x
    def minimum(self):return self.lo
    def setDecimals(self,x):self.dec=x
    def decimals(self):return self.dec


class Timer:
    def __init__(self):self.active=True
    def isActive(self):return self.active
    def stop(self):self.active=False
    def start(self):self.active=True


@contextmanager
def blocked(obj):
    old=obj.blockSignals(True)
    try:yield
    finally:obj.blockSignals(old)


def source_methods(path,class_name,names):
    module=ast.parse(path.read_text())
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods}==set(names)
    return methods


# All relevant callback implementations are extracted from the package itself.
METHODS=['_apply_program','_toggle_rf','_start_program','_stop_program','_toggle_pause',
         '_tick','_set_pause_at_end','_update_rf_button','_sync_rf_button']
methods=source_methods(ROOT/'ssrf_realtime/gui.py','Spin1RealtimeWindow',METHODS)
methods+=source_methods(ROOT/'ssrf_realtime/designer_gui.py','Spin1RealtimeWindow',['_design_tensor','_frozen'])
class_node=ast.ClassDef(name='ActualCallbacks',bases=[],keywords=[],body=methods,decorator_list=[])
module=ast.fix_missing_locations(ast.Module(body=[class_node],type_ignores=[]))
MESSAGES=[]
namespace=dict(np=np,blocked=blocked,contextmanager=contextmanager,
    PopulationSnapshot=PopulationSnapshot,PulsePlaybackController=PulsePlaybackController,
    QtWidgets=SimpleNamespace(QMessageBox=SimpleNamespace(
        warning=lambda *args:MESSAGES.append(args[1:]),critical=lambda *args:MESSAGES.append(args[1:]))))
exec(compile(module,'<actual GUI callback bodies>','exec'),namespace)
ActualCallbacks=namespace['ActualCallbacks']


class WindowHarness(ActualCallbacks):
    def __init__(self):
        self.model=IdealBinModel(IdealBinParams(n_bins=41,dt=.003,diffusion_enabled=False))
        self.paused=True;self.timer=Timer();self.steps_per_tick=12
        self.pause_at_program_end=False
        self.rf_button=Widget(callback=self._toggle_rf)
        self.pause_button=Widget(callback=lambda _:self._toggle_pause())
        self.gain_box=Widget();self.dt_box=Widget();self.material_label=Widget()
        self.pause_at_end_box=Widget(callback=self._set_pause_at_end)
        self.optimizer_settings=OptimizerSettings();self.material_name='Test material'
        self.trace=[];self.refreshes=0
    def rf_is_on(self):return self.rf_button.isChecked()
    def dnp_is_on(self):return False
    def _refresh_profile_picker(self):pass
    def _capture_material(self):return MaterialConfig.capture(self.model)
    def _start_new_trace(self,record_now=False):
        self.trace=[]
        if record_now:self._record_trace_point()
    def _record_trace_point(self):self.trace.append((self.model.t,self.model.n.copy()))
    def _update_plots(self):self.refreshes+=1;self._sync_rf_button()


def program():
    return PulseProgram(41,-3.,3.,profiles=[RFProfile('Generated/test',pulses=[
        BinPulse(14,2.,0.,.023),BinPulse(27,.6,0.,.023)])])


def test_exact_reported_flow_from_generated_install_to_RF_ON(monkeypatch):
    w=WindowHarness();before=w.model.n.copy()
    # The optimizer's result is supplied, but all load/start/tick callbacks are
    # the real source bodies. The mathematical optimizer is unchanged.
    def dialog(snapshot,settings,material,parent):
        return SimpleNamespace(install_requested=True,settings=settings,
            result=SimpleNamespace(program=program(),report={'recommended_playback_dt':.001}))
    monkeypatch.setitem(namespace,'TensorDesignerDialog',dialog)
    monkeypatch.setitem(namespace,'run_dialog',lambda d:0)
    w._design_tensor()
    assert w.paused and w.pause_at_program_end
    assert w.model.program_state=='ready' and not w.rf_is_on()
    assert w.model.params.dt==.001
    np.testing.assert_array_equal(before,w.model.n)
    # User presses Run simulation first; READY schedule must not run out.
    w.pause_button.click();w._tick();w._tick()
    assert not w.paused and w.model.program_state=='ready'
    assert not np.any(w.model.delivered_exposure)
    t=w.model.t
    # User's actual next action: RF ON, with no separate Start button.
    w.rf_button.click()
    assert w.model._epoch==t and w.model.params.rf_enabled
    assert w.model.program_state=='running'
    assert np.count_nonzero(w.model.applied_rf_field())==2
    while not w.paused:w._tick()
    assert w.model.program_state=='finished' and not w.rf_is_on()
    assert w.model.program_elapsed==pytest.approx(.023)
    assert w.model.delivered_exposure.sum()==pytest.approx(.023*2.6)
    assert np.max(abs(w.model.n-before))>1e-5
    assert 'finished' in w.rf_button.text and 'replay' in w.rf_button.text
    # Rendering synchronization did not trigger a second program automatically.
    assert w.rf_button.changes==1


def test_generic_example_starts_with_same_single_RF_button():
    w=WindowHarness();n=w.model.n.copy()
    w.rf_button.click();w._tick()
    assert not w.paused and w.model.program_state=='running'
    assert w.model.delivered_exposure.sum()>0
    assert np.max(abs(w.model.n-n))>1e-5


def test_loading_program_while_RF_on_does_not_start_the_replacement():
    w=WindowHarness();w.rf_button.click();w._tick()
    n=w.model.n.copy();t=w.model.t
    w._apply_program(program())
    assert not w.rf_is_on() and w.model.program_state=='ready'
    assert w.model._epoch is None
    assert w.model.t==t
    np.testing.assert_array_equal(w.model.n,n)
    w._tick()
    assert not np.any(w.model.delivered_exposure)


def test_zero_gain_shows_warning_and_resets_the_checkbox():
    w=WindowHarness();w.model.set_program_gain(0)
    start=len(MESSAGES)
    w.rf_button.click()
    assert not w.rf_is_on() and not w.model.params.rf_enabled
    assert w.model._epoch is None
    assert len(MESSAGES)==start+1
    assert 'gain is zero' in MESSAGES[-1][1]


def test_simulation_pause_does_not_consume_or_restart_pulse():
    w=WindowHarness();w.rf_button.click();w._tick()
    n=w.model.n.copy();t=w.model.t;epoch=w.model._epoch
    w._toggle_pause();w._tick()
    assert w.paused and w.model.t==t and w.model._epoch==epoch
    np.testing.assert_array_equal(w.model.n,n)
    w._toggle_pause();w._tick()
    assert w.model.t>t and w.model._epoch==epoch


def test_explicit_restart_still_works_without_double_start_signal():
    w=WindowHarness()
    w._start_program();w._tick();t=w.model.t
    w._start_program()
    assert w.model._epoch==t and not np.any(w.model.delivered_exposure)
    assert w.rf_button.changes==0


def test_muting_RF_preserves_running_clock_and_state():
    w=WindowHarness();w.rf_button.click();w._tick()
    epoch=w.model._epoch;exposure=w.model.delivered_exposure.copy()
    w.rf_button.click();w._tick();elapsed=w.model.program_elapsed
    np.testing.assert_array_equal(w.model.delivered_exposure,exposure)
    w.rf_button.click()
    assert w.model._epoch==epoch and w.model.program_elapsed==elapsed
