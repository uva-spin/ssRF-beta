"""Preview is isolated; explicit apply preserves AFP populations and RF budget."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import importlib.util
import numpy as np
import pytest
if not any(importlib.util.find_spec(m) for m in ('PyQt6','PySide6','PyQt5')):
    pytest.skip('No Qt binding installed',allow_module_level=True)
from ssrf_realtime.designer_gui import Spin1RealtimeWindow, QtWidgets
from ssrf_realtime.rf_response_gui import RFResponseDialog
from ssrf_realtime.ideal_model import IdealBinParams


def test_default_window_starts_tuned_with_discoverable_control():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win=Spin1RealtimeWindow();win.timer.stop();win.show();app.processEvents()
    try:
        assert win.model.params.rf_response_mode=='spectral'
        assert win.model.params.rf_peak_selectivity==.7
        assert win.model.params.rf_pedestal_gain==1.
        assert 'spin diffusion' in win.rf_response_button.text()
        assert win.model.program_state=='ready'
        assert win.controls_scroll.horizontalScrollBar().maximum()==0
        assert win.controls_scroll.viewport().width()>=win.controls_scroll.widget().minimumSizeHint().width()
    finally:win.close();app.processEvents()


def test_preview_and_apply_preserve_live_state_and_budget():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=31));win.timer.stop()
    win.mode_combo.setCurrentIndex(win.mode_combo.findData('combined'))
    win.afp_region_combo.setCurrentIndex(win.afp_region_combo.findData('selective'))
    win.afp_left_box.setValue(-1.7);win.afp_right_box.setValue(-1.4)
    win.pause_at_end_box.setChecked(True)
    win.model.afp_sweep([10]);win.optimizer_settings.max_total_exposure=.2
    before=win.model.n.copy();t=win.model.t;program=win.model.program.to_dict()
    d=RFResponseDialog(win._capture_material(),win)
    try:
        d.mode.setCurrentIndex(d.mode.findData('independent'))
        d.edits['rf_pedestal_gain'].setText('20')
        d.preview()
        assert d.value is None and win.model.params.rf_response_mode=='legacy'
        np.testing.assert_array_equal(win.model.n,before)
        d.commit();assert d.value is not None
        win._apply_rf_response_config(d.value)
        assert win.model.params.rf_pedestal_gain==20 and win.paused
        assert not win.model.params.rf_enabled
        assert win.optimizer_settings.max_total_exposure==.2
        np.testing.assert_array_equal(win.model.n,before)
        assert win.model.t==t and win.model.program.to_dict()==program
        assert win.model._afp_last_subset==[10]
        assert win.mode_combo.currentData()=='combined'
        assert win.afp_region_combo.currentData()=='selective'
        assert win.afp_left_box.value()==-1.7 and win.afp_right_box.value()==-1.4
        assert win.pause_at_program_end and win.pause_at_end_box.isChecked()
        assert win._capture_material().parameters['rf_response_mode']=='independent'
    finally:d.close();win.timer.stop();win.close();app.processEvents()


def test_switching_old_material_to_independent_keeps_its_density_response():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=31,capacity_rate_power=.6,
                                         capacity_rate_clip=7.));win.timer.stop()
    d=RFResponseDialog(win._capture_material(),win)
    try:
        d.mode.setCurrentIndex(d.mode.findData('independent'))
        model=d.candidate().make_model(preserve_state=False)
        np.testing.assert_array_equal(model.rf_rate_weights(),win.model.rf_rate_weights())
    finally:d.close();win.close();app.processEvents()


def test_spectral_RF_and_exchange_fields_apply_together_without_resetting_state():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win=Spin1RealtimeWindow(IdealBinParams(n_bins=61,rf_response_mode='spectral'));win.timer.stop()
    win.model.afp_sweep([19]);win.optimizer_settings.max_total_exposure=.3
    before=win.model.n.copy();program=win.model.program.to_dict()
    d=RFResponseDialog(win._capture_material(),win)
    try:
        assert d.tabs.count()==2
        d.edits['rf_spectral_strength'].setText('0.025')
        d.edits['rf_homogeneous_hwhm_R'].setText('0.02')
        d.edits['rf_peak_selectivity'].setText('0.6')
        d.edits['diffusion_scale'].setText('8')
        d.edits['zq_width_R'].setText('0.08')
        d.preview();assert d.value is None
        np.testing.assert_array_equal(win.model.n,before)
        d.commit();assert d.value is not None
        win._apply_rf_response_config(d.value)
        assert win.model.params.rf_spectral_strength==.025
        assert win.model.params.rf_homogeneous_hwhm_R==.02
        assert win.model.params.rf_peak_selectivity==.6
        assert win.model.params.diffusion_scale==8
        assert win.model.params.zq_width_R==.08
        assert win.optimizer_settings.max_total_exposure==.3
        assert win.model.program.to_dict()==program
        assert win.model._afp_last_subset==[19]
        np.testing.assert_array_equal(win.model.n,before)
        assert win.paused and not win.model.params.rf_enabled
    finally:d.close();win.timer.stop();win.close();app.processEvents()
