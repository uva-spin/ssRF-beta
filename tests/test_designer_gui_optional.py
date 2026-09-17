"""Exercise new windows and worker when a Qt binding is available.

QT_QPA_PLATFORM=offscreen pytest -q tests/test_designer_gui_optional.py
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import importlib.util
import time
import numpy as np
import pytest
if not any(importlib.util.find_spec(m) for m in ('PyQt6','PySide6','PyQt5')):
    pytest.skip('No Qt binding installed; new designer window not run here',allow_module_level=True)
from ssrf_realtime.designer_gui import (Spin1RealtimeWindow,TensorDesignerDialog,JSONEditor,QtWidgets)
from ssrf_realtime.ideal_model import IdealBinParams
from ssrf_realtime.material_config import MaterialConfig,PopulationSnapshot
from ssrf_realtime.tensor_optimizer import OptimizerSettings


@pytest.fixture(scope='module')
def app():return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_main_window_and_material_adoption(app):
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=31));win.timer.stop()
    n=win.model.n.copy();c=win._capture_material();c.parameters['diffusion_scale']=8
    win._adopt_model(c.make_model(win.model,True),c)
    assert win.paused
    assert win.tensor_button.text().startswith('Generate tensor')
    assert win.model.params.diffusion_scale==8
    np.testing.assert_array_equal(n,win.model.n)
    assert len(win.fig.axes)==4  # three principal axes + existing RF overlay axis
    win.close();app.processEvents()


def test_explicit_settings_editor_validation(app):
    from ssrf_realtime.ideal_model import IdealBinModel
    cfg=MaterialConfig.capture(IdealBinModel(IdealBinParams(n_bins=31)))
    d=JSONEditor('Test',cfg.to_dict(),MaterialConfig.from_dict)
    d.commit();assert d.value.parameters==cfg.parameters
    d.close()


def test_worker_preview_does_not_change_state(app):
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=31));win.timer.stop()
    n=win.model.n.copy();s=PopulationSnapshot.capture(win.model)
    o=OptimizerSettings(max_power=4,min_duration=.02,max_duration=.05,
        duration_samples=2,duration_refinements=0,max_iterations=4,starts=1,max_wall_seconds=30)
    d=TensorDesignerDialog(s,o,win._capture_material(),win)
    d._generate()
    until=time.monotonic()+30
    while d.worker.isRunning() and time.monotonic()<until:
        app.processEvents();time.sleep(.01)
    app.processEvents()
    assert not d.worker.isRunning()
    assert d.result is not None,d.log.toPlainText()
    assert d.table.rowCount()==d.result.report['active_bins']
    np.testing.assert_array_equal(win.model.n,n)
    d.close();win.close();app.processEvents()
