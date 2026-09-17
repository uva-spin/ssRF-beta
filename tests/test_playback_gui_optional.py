"""Real Qt signal/slot regression; requires an installed Qt binding.

Run: QT_QPA_PLATFORM=offscreen python -m pytest -q tests/test_playback_gui_optional.py
The generated-program test runs the actual designer worker and presses its
load button, then Run simulation and RF ON in the main window.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import importlib.util
import time
import numpy as np
import pytest
if not any(importlib.util.find_spec(x) for x in ('PyQt6','PySide6','PyQt5')):
    pytest.skip('No Qt binding: native playback button/worker tests not run',allow_module_level=True)
from ssrf_realtime import designer_gui
from ssrf_realtime.designer_gui import Spin1RealtimeWindow,QtWidgets
from ssrf_realtime.ideal_model import IdealBinParams
from ssrf_realtime.tensor_optimizer import OptimizerSettings


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_generic_example_rf_on_starts_without_separate_start(app):
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=41,diffusion_enabled=False));win.timer.stop()
    try:
        win.show();app.processEvents();before=win.model.n.copy()
        win.rf_button.click();win._tick()
        assert not win.paused and win.model.program_state=='running'
        assert win.model.delivered_exposure.sum()>0
        assert np.max(abs(win.model.n-before))>1e-6
    finally:
        win.timer.stop();win.close();app.processEvents()


def test_real_designer_calculate_load_run_then_RF_on(app,monkeypatch):
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=41,diffusion_enabled=False));win.timer.stop()
    win.optimizer_settings=OptimizerSettings(max_power=4,min_duration=.02,max_duration=.06,
        duration_samples=2,duration_refinements=0,max_iterations=5,starts=1,max_wall_seconds=30)
    before=win.model.n.copy()
    def run_real_dialog(d):
        d.show();app.processEvents();d.generate_button.click()
        deadline=time.monotonic()+45
        while d.worker.isRunning() and time.monotonic()<deadline:
            app.processEvents();time.sleep(.005)
        app.processEvents()
        assert not d.worker.isRunning(), 'Designer worker timed out'
        assert d.result is not None,d.log.toPlainText()
        assert d.install_button.isEnabled(),d.summary.text()
        d.install_button.click()
        assert d.install_requested
        return 1
    monkeypatch.setattr(designer_gui,'run_dialog',run_real_dialog)
    try:
        win._design_tensor();win.timer.stop()
        assert win.model.program_state=='ready' and not win.rf_is_on()
        assert win.paused and win.pause_at_end_box.isChecked()
        np.testing.assert_array_equal(win.model.n,before)
        win.pause_button.click();win._tick()
        assert win.model.program_state=='ready' and not win.model.delivered_exposure.any()
        win.rf_button.click()
        epoch=win.model._epoch
        while not win.paused:win._tick()
        assert win.model.program_state=='finished'
        assert not win.rf_is_on() and win.model.delivered_exposure.sum()>0
        assert win.model.program_elapsed==pytest.approx(win.model._compiled.end_time)
        assert win.model._epoch==epoch
        assert np.max(abs(win.model.n-before))>1e-6
        final=win.model.n.copy()
        # Completion must not start another run via programmatic setChecked.
        win._tick();np.testing.assert_array_equal(win.model.n,final)
        assert len(win.ax_trace.lines)==4
    finally:
        win.timer.stop();win.close();app.processEvents()


def test_three_panels_mirrors_and_material_state_reload(app):
    """Native Qt version of the Agg/source callback checks for the new views."""
    from ssrf_realtime.material_config import MaterialConfig
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=81,p0=.45)); win.timer.stop()
    try:
        win.show(); app.processEvents()
        assert len(win.fig.axes)==4
        assert len(win.ax_trace.lines)==4
        assert win.ax_spec.get_shared_x_axes().joined(win.ax_spec,win.ax_tensor)
        win.rf_button.click(); win._tick(); win.timer.stop()
        pair=win.model.pair_intensities(win.model.params.rf_burn_R)
        assert win.trace_Ip_minusR[-1]==pair['Iplus_minusR']
        assert win.trace_Im_minusR[-1]==pair['Iminus_minusR']
        np.testing.assert_array_equal(win.line_tensor.get_ydata(),
            np.asarray(win.line_Ip.get_ydata())-np.asarray(win.line_Im.get_ydata()))
        config=MaterialConfig.capture(win.model)
        copy=config.make_model(win.model,preserve_state=True)
        before=copy.n.copy()
        win._adopt_model(copy,config); win.timer.stop(); app.processEvents()
        assert len(win.trace_t)==len(win.trace_Ip_minusR)==len(win.trace_Im_minusR)==1
        np.testing.assert_array_equal(win.model.n,before)
        assert len(win.fig.axes)==4
        assert win.trace_line_Ip_minusR.get_linestyle()==':'
        assert win.trace_line_Im_minusR.get_linestyle()==':'
    finally:
        win.timer.stop(); win.close(); app.processEvents()
