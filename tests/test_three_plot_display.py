"""New-display regression tests; physics and playback use the shipped model."""
import sys
from pathlib import Path
from types import SimpleNamespace
import csv
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from ssrf_realtime.ideal_model import IdealBinParams
from ssrf_realtime.pulse_program import BinPulse, RFProfile, PulseProgram
from ssrf_realtime.playback import PulsePlaybackController
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gui_plot_harness import PlotHarness, FileDialog


def pulse(w, R=-.65, rate=2., duration=.08):
    j = int(np.argmin(abs(w.model.Rplus-R)))
    w.model.set_program(PulseProgram(len(w.model.Rplus), -3., 3., profiles=[
        RFProfile('Test pulse', pulses=[BinPulse(j, rate, 0., duration)])]))
    PulsePlaybackController(w.model).rf_on()


def histories(w):
    return (w.trace_Ip_R, w.trace_Im_R, w.trace_Ip_minusR, w.trace_Im_minusR)


def assert_current_point(w):
    v = w.model.pair_intensities(w.model.params.rf_burn_R)
    for h, key in zip(histories(w), ('Iplus_R', 'Iminus_R', 'Iplus_minusR', 'Iminus_minusR')):
        assert h[-1] == v[key]
        assert len(h) == len(w.trace_t)


def test_three_rows_and_shared_frequency_axis():
    w = PlotHarness()
    assert len(w.fig.axes) == 4  # 3 rows + existing RF right axis
    assert len(w.ax_trace.lines) == 4
    assert w.ax_spec.get_shared_x_axes().joined(w.ax_spec, w.ax_tensor)
    assert not w.ax_spec.get_shared_x_axes().joined(w.ax_spec, w.ax_trace)
    w.canvas.draw()
    assert w.ax_spec.get_position().y0 > w.ax_tensor.get_position().y1
    assert w.ax_tensor.get_position().y0 > w.ax_trace.get_position().y1
    assert w.ax_spec.get_xlim() == w.ax_tensor.get_xlim()
    w.ax_spec.set_xlim(-1.3, .7)
    assert w.ax_tensor.get_xlim() == (-1.3, .7)
    assert w.ax_trace.get_xlim() != (-1.3, .7)


def test_requested_mirror_styles_and_meanings():
    w = PlotHarness()
    assert w.trace_line_Ip_R.get_linestyle() == '-'
    assert w.trace_line_Im_R.get_linestyle() == '-'
    assert w.trace_line_Ip_minusR.get_linestyle() == ':'
    assert w.trace_line_Im_minusR.get_linestyle() == ':'
    assert to_rgba(w.trace_line_Ip_minusR.get_color()) == to_rgba('tab:blue')
    assert to_rgba(w.trace_line_Im_minusR.get_color()) == to_rgba('goldenrod')
    assert '-M' in w.trace_line_Ip_minusR.get_label()
    assert '-M' in w.trace_line_Im_minusR.get_label()


@pytest.mark.parametrize('P', [0., .1, .45, .58, -.1, -.58])
def test_tensor_is_literal_signed_subtraction_and_integrates_correctly(P):
    w = PlotHarness(IdealBinParams(n_bins=101, p0=P))
    R, a, b, _ = w.model.spectrum(noise_sigma=0)
    np.testing.assert_array_equal(w.line_tensor.get_xdata(), R)
    np.testing.assert_array_equal(w.line_tensor.get_ydata(), a-b)
    # Integral convention matches the existing discrete population normalization.
    Q = np.sum(w.line_tensor.get_ydata()) * w.model.dR / w.model.display_cal
    assert Q == pytest.approx(w.model.polarizations()['Q'], abs=5e-13)
    lo, hi = w.ax_tensor.get_ylim()
    assert lo <= min(0., np.min(a-b)) and hi >= max(0., np.max(a-b))


@pytest.mark.parametrize('P,R', [(.45,-.9),(.45,.4),(-.58,-.9),(-.58,.6),(.45,0.)])
def test_recording_start_burn_and_recovery_all_four_use_live_populations(P,R):
    w = PlotHarness(IdealBinParams(n_bins=81, p0=P, rf_burn_R=R, dt=.002))
    assert w.trace_t == [0.]
    assert_current_point(w)
    pulse(w, R=R)
    for _ in range(8):
        w.model.step(n_steps=5)
        w._record_trace_point()
        assert_current_point(w)
    w.model.stop_program()
    for _ in range(4):
        w.model.step(n_steps=5)
        w._record_trace_point()
        assert_current_point(w)
    w._update_plots()
    for artist, h in zip((w.trace_line_Ip_R, w.trace_line_Im_R,
                         w.trace_line_Ip_minusR, w.trace_line_Im_minusR), histories(w)):
        np.testing.assert_array_equal(artist.get_xdata(), w.trace_t)
        np.testing.assert_array_equal(artist.get_ydata(), h)
    a = np.asarray(w.line_Ip.get_ydata()); b = np.asarray(w.line_Im.get_ydata())
    np.testing.assert_array_equal(w.line_tensor.get_ydata(), a-b)
    lo, hi = w.ax_trace.get_ylim()
    assert lo < min(min(h) for h in histories(w))
    assert hi > max(max(h) for h in histories(w))


def test_tensor_reuses_same_display_noise_as_upper_branches():
    w = PlotHarness(IdealBinParams(n_bins=101, noise_sigma=.01))
    original = w.model.spectrum
    calls = []
    def wrapped(**kw):
        calls.append(kw)
        return original(**kw)
    w.model.spectrum = wrapped
    n = w.model.n.copy(); t = w.model.t
    w._update_plots()
    assert len(calls) == 1
    a, b = w.line_Ip.get_ydata(), w.line_Im.get_ydata()
    np.testing.assert_array_equal(w.line_tensor.get_ydata(), a-b)
    np.testing.assert_array_equal(w.line_total.get_ydata(), a+b)
    np.testing.assert_array_equal(w.model.n, n)
    assert w.model.t == t
    # Monitoring remains population-derived, just as the earlier direct traces.
    assert_current_point(w)


def test_monitor_changes_restart_all_histories_without_touching_RF():
    w = PlotHarness(); pulse(w)
    w.model.step(n_steps=7); w._record_trace_point()
    n = w.model.n.copy(); t = w.model.t; epoch = w.model._epoch
    u = w.model.applied_rf_field().copy()
    w._set_R(-.91)
    assert w.trace_t == [0.]
    assert_current_point(w)
    np.testing.assert_array_equal(w.model.n, n)
    np.testing.assert_array_equal(w.model.applied_rf_field(), u)
    assert w.model.t == t and w.model._epoch == epoch
    R = w.model.params.rf_burn_R
    assert w.tensor_monitor_line.get_xdata() == [R,R]
    assert w.tensor_mirror_line.get_xdata() == [-R,-R]
    # Rounding to same monitor does not erase the trace.
    w.model.step(n_steps=1); w._record_trace_point()
    w._set_R(R)
    assert len(w.trace_t) == 2


def test_restarting_trace_and_replacing_model_clears_old_mirror_samples():
    w = PlotHarness(); pulse(w)
    w.model.step(n_steps=10); w._record_trace_point()
    w._start_new_trace(record_now=True)
    assert w.trace_t == [0.]; assert_current_point(w)
    from ssrf_realtime.ideal_model import IdealBinModel
    w.model = IdealBinModel(IdealBinParams(n_bins=61,p0=-.58,rf_burn_R=-.7))
    # Equivalent history lifecycle used by designer._adopt_model.
    w.trace_t = []; w.trace_Ip_R = []; w.trace_Im_R = []
    w._start_new_trace(record_now=True); w._update_plots()
    assert w.trace_t == [0.]; assert_current_point(w)
    assert w.trace_Ip_minusR[0] < 0 and w.trace_Im_minusR[0] < 0
    assert len(w.line_tensor.get_xdata()) == 61


def test_trace_length_limit_keeps_all_series_in_step():
    w = PlotHarness(); w.trace_max_points = 4
    pulse(w)
    for _ in range(9):
        w.model.step(n_steps=1); w._record_trace_point()
    assert len(w.trace_t) == 4
    assert all(len(h) == 4 for h in histories(w))
    assert_current_point(w)
    w._update_plots()


def test_RF_endpoint_pause_updates_tensor_and_mirror_once_then_recovery():
    w = PlotHarness(IdealBinParams(n_bins=81, dt=.003)); pulse(w, duration=.017)
    controller = PulsePlaybackController(w.model)
    update = controller.advance(100, pause_at_end=True)
    assert update.pause_requested and update.completed
    w._record_trace_point(); w._update_plots()
    assert_current_point(w)
    assert w.model.program_elapsed == pytest.approx(.017)
    n = w.model.n.copy(); expo = w.model.delivered_exposure.copy()
    for _ in range(4): w._update_plots()
    np.testing.assert_array_equal(w.model.n,n)
    np.testing.assert_array_equal(w.model.delivered_exposure,expo)
    controller.advance(10,pause_at_end=False)
    w._record_trace_point(); w._update_plots()
    assert_current_point(w)
    assert not w.model.params.rf_enabled


def test_export_appends_mirror_columns_preserving_legacy_prefix(tmp_path):
    w = PlotHarness(); pulse(w)
    w.model.step(n_steps=10); w._record_trace_point()
    FileDialog.next_path = tmp_path/'monitor.csv'
    n = w.model.n.copy(); t=w.model.t
    w._export_results()
    with FileDialog.next_path.open() as f: rows=list(csv.reader(f))
    assert rows[0][:5] == ['time_since_selection','simulation_time','monitor_R','Iplus','Iminus']
    assert rows[0][5:] == ['mirror_R','Iplus_mirror','Iminus_mirror']
    assert len(rows)-1 == len(w.trace_t)
    for row,k in zip(rows[1:],range(len(w.trace_t))):
        assert float(row[5]) == -w.trace_start_R
        assert float(row[6]) == w.trace_Ip_minusR[k]
        assert float(row[7]) == w.trace_Im_minusR[k]
    assert (tmp_path/'monitor_exposure.csv').exists()
    assert w.timer.isActive()
    np.testing.assert_array_equal(w.model.n,n); assert w.model.t==t


def test_display_alone_does_not_generate_physics_or_consume_pulses():
    w = PlotHarness(); pulse(w)
    n = w.model.n.copy(); t = w.model.t; epoch = w.model._epoch
    u = w.model.applied_rf_field().copy()
    for _ in range(7):
        w._record_trace_point(); w._update_plots()
    np.testing.assert_array_equal(w.model.n,n)
    np.testing.assert_array_equal(w.model.applied_rf_field(),u)
    assert w.model.t==t and w.model._epoch==epoch
    assert not np.any(w.model.delivered_exposure)


def test_plot_clicks_do_not_move_markers():
    w=PlotHarness(); R=w.model.params.rf_burn_R
    for ax in (w.ax_spec,w.ax_tensor,w.ax_trace):
        w._on_spectrum_click(SimpleNamespace(inaxes=ax,xdata=-1.3))
    assert w.model.params.rf_burn_R == R


def test_zero_tensor_autoscaling_does_not_accumulate_or_blow_up():
    w=PlotHarness(IdealBinParams(n_bins=51,p0=0))
    bounds=w.ax_tensor.get_ylim()
    for _ in range(40): w._update_plots()
    assert w.ax_tensor.get_ylim()==bounds
    assert max(abs(v) for v in bounds) <= 1e-6
