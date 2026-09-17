"""Optional real Qt smoke tests; run with QT_QPA_PLATFORM=offscreen.

These are skipped when no Qt binding is installed. Compilation and headless
model tests are NOT substitutes for this test on the user's desktop platform.
"""
import os
import importlib.util
import numpy as np
import pytest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
if not any(importlib.util.find_spec(x) is not None for x in ('PyQt6','PySide6','PyQt5')):
    pytest.skip('No Qt binding installed: GUI runtime test not executed',allow_module_level=True)
from ssrf_realtime.gui import Spin1RealtimeWindow, QtWidgets
from ssrf_realtime.ideal_model import IdealBinParams
from ssrf_realtime.profile_editor import ProfileEditor, CHECK


def test_window_program_monitoring_and_editor():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=101,diffusion_enabled=False))
    win.timer.stop()
    try:
        win.show();app.processEvents()
        assert len(win.ax_trace.lines)==4
        assert len(win.fig.axes)==4   # three panels, including the top right RF axis
        # Explicit start arms the program without changing the population state.
        n=win.model.n.copy();win._start_program()
        np.testing.assert_array_equal(n,win.model.n)
        win.timer.stop();win._tick()
        assert win.model.t>0
        u=win.model.applied_rf_field();epoch=win.model._epoch
        win.R_box.setValue(-0.9);app.processEvents()
        np.testing.assert_array_equal(u,win.model.applied_rf_field())
        assert epoch==win.model._epoch
        editor=ProfileEditor(win.model.program,win.model.params.rf_burn_R,win)
        editor._example()
        assert len(editor.program.profiles)==2
        assert editor.table.rowCount()>2
        editor.table.item(1,3).setText('3.25')
        assert editor.program.profiles[0].pulses[1].rate==3.25
        editor.table.item(1,4).setText('0.031')
        editor.table.item(1,5).setText('0.007')
        assert float(editor.table.item(1,6).text())==pytest.approx(0.038)
        editor._add_profile();editor._generate()
        assert len(editor.program.profiles)==3
        assert len(editor.program.profiles[-1].pulses)>0
        editor._apply()
        win._apply_program(editor.program)
        assert win.model.program_state=='ready'
        assert not win.rf_is_on()
        editor.close()
    finally:
        win.timer.stop();win.close();app.processEvents()
