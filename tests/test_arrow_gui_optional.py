"""Actual Qt event tests; skipped only if no Qt binding is installed.

Run locally with QT_QPA_PLATFORM=offscreen python -m pytest -q.
"""
import os
import importlib
import importlib.util
from types import SimpleNamespace
import numpy as np
import pytest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
if not any(importlib.util.find_spec(x) for x in ('PyQt6','PySide6','PyQt5')):
    pytest.skip('Qt unavailable: arrow/marker/editor GUI tests not executed',allow_module_level=True)
from ssrf_realtime.baseline_gui import QT_BINDING,QtCore,QtWidgets
from ssrf_realtime.safe_widgets import ArrowDoubleSpinBox,ArrowSpinBox,NoWheelComboBox
from ssrf_realtime.gui import Spin1RealtimeWindow
from ssrf_realtime.profile_editor import ProfileEditor
from ssrf_realtime.ideal_model import IdealBinParams
QtGui=importlib.import_module(QT_BINDING+'.QtGui')
QtTest=importlib.import_module(QT_BINDING+'.QtTest')
QT=QtCore.Qt
KEY=QT.Key if hasattr(QT,'Key') else QT
BUTTON=QT.MouseButton if hasattr(QT,'MouseButton') else QT
MOD=QT.KeyboardModifier if hasattr(QT,'KeyboardModifier') else QT
PHASE=QT.ScrollPhase if hasattr(QT,'ScrollPhase') else QT
STYLE=QtWidgets.QStyle
SUB=STYLE.SubControl if hasattr(STYLE,'SubControl') else STYLE
COMPLEX=STYLE.ComplexControl if hasattr(STYLE,'ComplexControl') else STYLE


@pytest.fixture(scope='module')
def app():
    application=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application


def wheel(widget):
    pos=QtCore.QPointF(10,10)
    event=QtGui.QWheelEvent(pos,pos,QtCore.QPoint(),QtCore.QPoint(0,120),
                           BUTTON.NoButton,MOD.NoModifier,PHASE.NoScrollPhase,False)
    QtWidgets.QApplication.sendEvent(widget,event)


@pytest.mark.parametrize('factory',[ArrowDoubleSpinBox,ArrowSpinBox])
def test_only_visible_arrows_step_values(app,factory):
    box=factory();box.setRange(0,100);box.setValue(5);box.resize(170,35);box.show();app.processEvents()
    try:
        box.setFocus();before=box.value()
        wheel(box);wheel(box.lineEdit())
        QtTest.QTest.keyClicks(box.lineEdit(),'123')
        QtTest.QTest.keyClick(box,KEY.Key_Up)
        QtTest.QTest.keyClick(box,KEY.Key_Down)
        assert box.value()==before
        option=QtWidgets.QStyleOptionSpinBox();box.initStyleOption(option)
        rect=box.style().subControlRect(COMPLEX.CC_SpinBox,option,SUB.SC_SpinBoxUp,box)
        QtTest.QTest.mouseClick(box,BUTTON.LeftButton,pos=rect.center());app.processEvents()
        assert box.value()==before+box.singleStep()
    finally:box.close()


def test_dropdown_not_changed_by_scroll(app):
    box=NoWheelComboBox();box.addItems(['first','second','third']);box.setCurrentIndex(1)
    wheel(box);assert box.currentIndex()==1


def test_monitor_markers_do_not_move_RF_or_restart_clock(app):
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=101,diffusion_enabled=False))
    win.timer.stop()
    try:
        win.show();app.processEvents();win._start_program();win.timer.stop();win.model.step(3)
        state=win.model.n.copy();epoch=win.model._epoch;t=win.model.t
        field=win.model.applied_rf_field().copy()
        initial=win.model.params.rf_burn_R
        win._on_spectrum_click(SimpleNamespace(inaxes=win.ax_spec,xdata=-.9))
        assert win.model.params.rf_burn_R==initial
        wheel(win.R_box);assert win.model.params.rf_burn_R==initial
        win.monitor_bin_box.setValue(20);win._update_plots()
        assert win.model.params.rf_burn_R==win.model.Rplus[20]
        assert list(win.burn_line.get_xdata())==[win.model.Rplus[20]]*2
        assert list(win.mirror_line.get_xdata())==[-win.model.Rplus[20]]*2
        assert float(win.mirror_R_readout.text())==pytest.approx(-win.model.Rplus[20])
        assert len(win.ax_trace.lines)==4
        np.testing.assert_array_equal(state,win.model.n)
        np.testing.assert_array_equal(field,win.model.applied_rf_field())
        assert win.model._epoch==epoch and win.model.t==t
        assert all(isinstance(x,(ArrowDoubleSpinBox,ArrowSpinBox))
                   for x in win.findChildren(QtWidgets.QAbstractSpinBox))
    finally:win.timer.stop();win.close();app.processEvents()


def test_every_bin_has_independent_numeric_editors(app):
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=101,diffusion_enabled=False));win.timer.stop()
    editor=ProfileEditor(win.model.program,parent=win)
    try:
        editor._example();p=editor._profile();old=[vars(x).copy() for x in p.pulses]
        for row in range(editor.table.rowCount()):
            for col in (3,4,5):assert isinstance(editor.table.cellWidget(row,col),ArrowDoubleSpinBox)
        editor.apply_exact_values([1],rate=3.45678912345,start=.0003,duration=.0007)
        p=editor._profile()
        assert p.pulses[1].rate==3.45678912345 and p.pulses[1].duration==.0007
        assert vars(p.pulses[0])==old[0]
        editor.table.cellWidget(1,3).setValue(2.75)
        assert p.pulses[1].rate==2.75
        before=p.pulses[1].rate;wheel(editor.table.cellWidget(1,3));assert p.pulses[1].rate==before
        editor._add_profile();editor.left_R.setValue(-.3);editor.right_R.setValue(.3);editor._ensure_bins()
        assert len(editor._profile().pulses)==11
        assert all(x.rate==0 for x in editor._profile().pulses)
        before=win.model.program.to_dict();editor.reject()
        assert win.model.program.to_dict()==before
    finally:editor.close();win.timer.stop();win.close();app.processEvents()
