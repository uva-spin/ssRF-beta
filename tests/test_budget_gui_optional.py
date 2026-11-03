"""Optional Qt budget control, worker and material round-trip checks."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import importlib.util
import time
from dataclasses import asdict

import numpy as np
import pytest

if not any(importlib.util.find_spec(m) for m in ('PyQt6','PySide6','PyQt5')):
    pytest.skip('No Qt binding installed',allow_module_level=True)

from ssrf_realtime.designer_gui import Spin1RealtimeWindow, TensorDesignerDialog, QtWidgets
from ssrf_realtime.ideal_model import IdealBinParams
from ssrf_realtime.material_config import MaterialConfig, PopulationSnapshot
from ssrf_realtime.tensor_optimizer import OptimizerSettings


def test_budget_controls_worker_and_material_persistence(tmp_path):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=31))
    win.timer.stop()
    win.model.afp_sweep([10])
    before=win.model.n.copy()
    settings=OptimizerSettings(max_power=4,min_duration=.02,max_duration=.05,
        duration_samples=2,duration_refinements=0,max_iterations=10,starts=1,max_wall_seconds=30)
    dialog=TensorDesignerDialog(PopulationSnapshot.capture(win.model),settings,win._capture_material(),win)
    try:
        assert not dialog.exposure_budget_check.isChecked()
        assert not dialog.exposure_budget_box.isEnabled()
        assert dialog._read_settings().max_total_exposure is None
        assert not dialog.total_power_check.isChecked()
        dialog.total_power_check.setChecked(True)
        dialog.total_power_box.setValue(1.)
        assert dialog._read_settings().max_total_power == 1.
        dialog.profile_width_box.setValue(.03)
        assert dialog._read_settings().profile_smoothing_fwhm_R == .03
        assert dialog.controls['max_steps'].value()==settings.max_steps
        dialog.controls['max_steps'].setValue(12000)
        assert dialog._read_settings().max_steps==12000
        dialog.exposure_budget_check.setChecked(True)
        dialog.exposure_budget_box.setValue(.2)
        assert dialog._read_settings().max_total_exposure == .2
        dialog._generate()
        assert not dialog.exposure_budget_box.isEnabled()
        assert not dialog.total_power_box.isEnabled()
        assert not dialog.profile_width_box.isEnabled()
        until=time.monotonic()+30
        while dialog.worker.isRunning() and time.monotonic()<until:
            app.processEvents(); time.sleep(.01)
        app.processEvents()
        assert not dialog.worker.isRunning()
        assert dialog.result is not None,dialog.log.toPlainText()
        assert dialog.result.report['integrated_base_exposure'] <= .2*(1+1e-12)
        assert dialog.result.powers.sum() <= 1.*(1+1e-12)
        assert 'Total RF exposure:' in dialog.summary.text()
        np.testing.assert_array_equal(win.model.n,before)
        config=win._capture_material();config.optimizer=asdict(dialog.settings)
        config.save(tmp_path/'material.json')
        loaded=MaterialConfig.load(tmp_path/'material.json')
        assert loaded.optimizer['max_steps']==12000
        win._adopt_model(loaded.make_model(win.model,True),loaded)
        assert win.optimizer_settings.max_total_exposure == .2
        assert win.optimizer_settings.max_total_power == 1.
        assert win.optimizer_settings.profile_smoothing_fwhm_R == .03
        used=dialog.result.report['integrated_base_exposure']
        dialog._use_result_budget()
        assert dialog.exposure_budget_check.isChecked()
        assert dialog._read_settings().max_total_exposure == pytest.approx(used,abs=1e-10)
        assert dialog.result is None and not dialog.install_button.isEnabled()
        dialog.exposure_budget_check.setChecked(False)
        assert dialog._read_settings().max_total_exposure is None
    finally:
        if dialog.worker and dialog.worker.isRunning():
            dialog.worker.cancel_event.set(); dialog.worker.wait(5000)
        dialog.close();win.timer.stop();win.close();app.processEvents()
