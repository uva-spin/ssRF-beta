"""Compact spectrum / tensor / monitor view for the ideal-bin RF simulator.

All displayed curves are projections of the existing populations. This module
does not change RF commands, population derivatives or program timing.
"""
from __future__ import annotations
import sys
import csv
from pathlib import Path
from typing import Optional
import numpy as np

from .baseline_gui import Spin1RealtimeWindow as BaselineWindow, QtCore, QtWidgets
from .ideal_model import IdealBinModel, IdealBinParams
from .model import Spin1Params
from .pulse_program import PulseProgram
from .playback import PulsePlaybackController
from .profile_editor import ProfileEditor, blocked
from .safe_widgets import ArrowDoubleSpinBox, ArrowSpinBox, NoWheelComboBox
from .profile_editing import nearest_bin
from .numeric_dialogs import ExactValuesDialog, accepted, run_dialog


class Spin1RealtimeWindow(BaselineWindow):
    def __init__(self, params: Optional[Spin1Params] = None):
        # Same plot/population conventions; construct the ideal-control subclass,
        # not a rewritten spin model.
        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle("Real-time ss-RF — custom bins / Boltzmann recovery")
        self.model = IdealBinModel(params or IdealBinParams())
        monitor = nearest_bin(self.model.Rplus, self.model.params.rf_burn_R)
        self.model.params.rf_burn_R = float(self.model.Rplus[monitor])
        self.trace_t0 = self.model.t
        self.trace_t = []
        self.trace_Ip_R = []
        self.trace_Im_R = []
        self.trace_Ip_minusR = []
        self.trace_Im_minusR = []
        self.trace_max_points = 4500
        self.trace_start_R = float(self.model.params.rf_burn_R)
        self.trace_start_vals = {}
        self._build_ui()
        self._init_plots()
        self._start_new_trace(record_now=True)
        self._update_plots()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(35)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.resize(1000, 555)

    def _spin_box(self, label, value, lo, hi, step, decimals, callback):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        box = ArrowDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setValue(value)
        box.valueChanged.connect(callback)
        row.addWidget(box)
        return row, box

    def _int_box(self, label, value, lo, hi, step, callback):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        box = ArrowSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setValue(value)
        box.valueChanged.connect(callback)
        row.addWidget(box)
        return row, box

    def _build_ui(self):
        # Keep all existing material, diffusion, DNP, T1 and initial-state controls.
        super()._build_ui()
        old_group = self.R_box.parentWidget()
        parent_layout = old_group.parentWidget().layout()
        position = parent_layout.indexOf(old_group)
        parent_layout.removeWidget(old_group)
        old_group.hide()
        old_group.deleteLater()

        markers = QtWidgets.QGroupBox("Marker positions — monitoring only")
        markers_layout = QtWidgets.QVBoxLayout(markers)
        k = nearest_bin(self.model.Rplus, self.model.params.rf_burn_R)
        row, self.monitor_bin_box = self._int_box("Monitor bin M (0-based)", k,
                                                  0, len(self.model.Rplus)-1, 1,
                                                  self._set_monitor_bin)
        markers_layout.addLayout(row)
        row, self.R_box = self._spin_box("Monitor R (M)", self.model.params.rf_burn_R,
                                        self.model.params.r_min, self.model.params.r_max,
                                        self.model.dR, 9, self._set_R)
        markers_layout.addLayout(row)
        self.R_box.setToolTip("Use arrows to move the monitor one bin. This does NOT move any RF pulse.")
        mirror_row = QtWidgets.QHBoxLayout()
        mirror_row.addWidget(QtWidgets.QLabel("Mirror R = -M"))
        self.mirror_R_readout = QtWidgets.QLineEdit()
        self.mirror_R_readout.setReadOnly(True)
        mirror_row.addWidget(self.mirror_R_readout)
        markers_layout.addLayout(mirror_row)
        self.monitor_exact_button = QtWidgets.QPushButton("Enter exact monitor R...")
        self.monitor_exact_button.clicked.connect(self._enter_monitor_R)
        markers_layout.addWidget(self.monitor_exact_button)
        hint = QtWidgets.QLabel("M chooses the solid monitor traces; -M supplies the dotted mirror traces. "
                               "Plot clicks and scrolling do not move markers or alter settings.")
        hint.setWordWrap(True); hint.setStyleSheet("font-size: 10px;")
        markers_layout.addWidget(hint)
        parent_layout.insertWidget(position, markers)

        group = QtWidgets.QGroupBox("RF program — custom per-bin commands")
        layout = QtWidgets.QVBoxLayout(group)
        row, self.gain_box = self._spin_box("global RF gain", self.model.program.gain,
                                           0.0, 1000.0, 0.1, 4, self._set_gain)
        layout.addLayout(row)
        self.edit_button = QtWidgets.QPushButton("Edit EVERY bin: power / start / duration")
        self.edit_button.clicked.connect(self._edit_program)
        self.start_button = QtWidgets.QPushButton("Start / restart RF program")
        self.start_button.clicked.connect(self._start_program)
        self.stop_button = QtWidgets.QPushButton("Stop program (keep evolving)")
        self.stop_button.clicked.connect(self._stop_program)
        for b in (self.edit_button, self.start_button, self.stop_button): layout.addWidget(b)
        self.pause_at_end_box = QtWidgets.QCheckBox("Pause at program end")
        self.pause_at_end_box.setChecked(bool(getattr(self, "pause_at_program_end", False)))
        self.pause_at_end_box.setToolTip(
            "Hold the populations exactly at the final pulse endpoint. "
            "Run simulation then continues RF-off recovery. No populations are reset.")
        self.pause_at_end_box.toggled.connect(self._set_pause_at_end)
        layout.addWidget(self.pause_at_end_box)
        io = QtWidgets.QHBoxLayout()
        for text, fun in (("Load program",self._load_program),("Save program",self._save_program)):
            b = QtWidgets.QPushButton(text); b.clicked.connect(fun); io.addWidget(b)
        layout.addLayout(io)
        export = QtWidgets.QPushButton("Export trace / exposure CSV")
        export.clicked.connect(self._export_results)
        layout.addWidget(export)
        self.program_label = QtWidgets.QLabel()
        self.program_label.setWordWrap(True)
        layout.addWidget(self.program_label)
        self.profile_view_combo = NoWheelComboBox()
        self.profile_view_combo.setObjectName("profile_bounds_selector")
        layout.addWidget(QtWidgets.QLabel("Show RF-profile limits L / R"))
        layout.addWidget(self.profile_view_combo)
        self.profile_bounds_label = QtWidgets.QLabel()
        self.profile_bounds_label.setWordWrap(True)
        self.profile_bounds_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.profile_bounds_label)
        self._refresh_profile_picker()
        self.profile_view_combo.currentIndexChanged.connect(self._profile_view_changed)
        hint = QtWidgets.QLabel(
            "RF positions come from the pulse-table bins, not M. "
            "L/R show the configured extent of the selected profile. "
            "RF ON starts a READY or FINISHED program at program t=0. "
            "During a run, RF OFF mutes RF while "
            "the schedule advances; Pause freezes both. Recovery continues after pulses end."
        )
        hint.setWordWrap(True); hint.setStyleSheet("font-size: 10px;")
        layout.addWidget(hint)
        parent_layout.insertWidget(position+1, group)
        self.info_label.setStyleSheet("font-size: 10px;")

    def _init_plots(self):
        # Keep the existing compact window height. The two spectral panels
        # share a physical-R axis; the monitor panel has its own time axis.
        grid = self.fig.add_gridspec(3, 1, height_ratios=[1.4, 0.9, 1.1])
        self.ax_spec = self.fig.add_subplot(grid[0, 0])
        self.ax_tensor = self.fig.add_subplot(grid[1, 0], sharex=self.ax_spec)
        self.ax_trace = self.fig.add_subplot(grid[2, 0])
        self.line_Ip, = self.ax_spec.plot([],[],drawstyle="steps-mid",label="I+(R)")
        self.line_Im, = self.ax_spec.plot([],[],drawstyle="steps-mid",label="I-(R)")
        self.line_total, = self.ax_spec.plot([],[],linewidth=1.2,label="total")
        self.burn_line = self.ax_spec.axvline(self.model.params.rf_burn_R,linestyle="--",linewidth=0.8,label="monitor M")
        self.mirror_line = self.ax_spec.axvline(-self.model.params.rf_burn_R,linestyle=":",linewidth=0.8,label="mirror -M")
        self.point_Ip, = self.ax_spec.plot([],[],marker="o",linestyle="None",markersize=4,label="_nolegend_")
        self.point_Im, = self.ax_spec.plot([],[],marker="s",linestyle="None",markersize=4,label="_nolegend_")
        self.point_Ip_m, = self.ax_spec.plot([],[],marker="^",linestyle="None",markersize=4,label="_nolegend_")
        self.point_Im_m, = self.ax_spec.plot([],[],marker="v",linestyle="None",markersize=4,label="_nolegend_")
        self.profile_left_line = self.ax_spec.axvline(0, linestyle="-.", linewidth=0.8,
                                                      alpha=0.55, label="profile bounds L/R")
        self.profile_right_line = self.ax_spec.axvline(0, linestyle="-.", linewidth=0.8,
                                                       alpha=0.55, label="_nolegend_")
        self.profile_left_line.set_visible(False); self.profile_right_line.set_visible(False)
        self.monitor_text = self.ax_spec.text(0, 0.99, "M", transform=self.ax_spec.get_xaxis_transform(),
                                              ha="center", va="top", fontsize=7)
        self.mirror_text = self.ax_spec.text(0, 0.91, "-M", transform=self.ax_spec.get_xaxis_transform(),
                                             ha="center", va="top", fontsize=7)
        self.left_text = self.ax_spec.text(0, 0.82, "L", transform=self.ax_spec.get_xaxis_transform(),
                                          ha="center", va="top", fontsize=7)
        self.right_text = self.ax_spec.text(0, 0.74, "R", transform=self.ax_spec.get_xaxis_transform(),
                                           ha="center", va="top", fontsize=7)
        self.ax_rf_profile = self.ax_spec.twinx()
        self.line_preview, = self.ax_rf_profile.plot([],[],linestyle=":",linewidth=0.9,alpha=0.5,
                                                    drawstyle="steps-mid",label="program envelope")
        self.line_rf_profile, = self.ax_rf_profile.plot([],[],linestyle="--",linewidth=1.3,
                                                       drawstyle="steps-mid",label="RF applied now")
        self.ax_rf_profile.set_ylabel("RF base rate U",fontsize=8)
        self.ax_rf_profile.tick_params(axis="y",labelsize=7)
        # Label the common frequency axis on the middle panel, not twice.
        self.ax_spec.tick_params(axis="x", labelbottom=False)
        self.ax_spec.set_ylabel("intensity [arb.]",fontsize=9)
        self.ax_spec.set_title("Overlapping Pake doublet and ideal RF commands",fontsize=10)
        handles, labels = self.ax_spec.get_legend_handles_labels()
        h2,l2 = self.ax_rf_profile.get_legend_handles_labels()
        self.ax_spec.legend(handles+h2,labels+l2,loc="upper left",ncol=3,fontsize=7)
        self.line_tensor, = self.ax_tensor.plot(
            [], [], linewidth=1.3, drawstyle="steps-mid", label="I+(R) - I-(R)")
        self.tensor_zero = self.ax_tensor.axhline(
            0.0, linestyle="--", linewidth=0.7, alpha=0.6, label="_nolegend_")
        self.tensor_monitor_line = self.ax_tensor.axvline(
            self.model.params.rf_burn_R, linestyle="--", linewidth=0.8, alpha=0.7)
        self.tensor_mirror_line = self.ax_tensor.axvline(
            -self.model.params.rf_burn_R, linestyle=":", linewidth=0.8, alpha=0.7)
        self.ax_tensor.set_title("Tensor spectrum: I+(R) - I-(R)", fontsize=9)
        self.ax_tensor.set_xlabel("physical R", fontsize=8)
        self.ax_tensor.set_ylabel("I+ - I- [arb.]", fontsize=8)

        self.trace_line_Ip_R, = self.ax_trace.plot(
            [], [], label="I+(M,t) monitor", linewidth=1.3)
        self.trace_line_Im_R, = self.ax_trace.plot(
            [], [], label="I-(M,t) monitor", linewidth=1.3)
        # User-requested colors: blue for the + mirror, yellow for the - mirror.
        # A dark yellow remains readable on a white canvas. Neither trace is
        # constructed by scaling/reflection of a hole; both use live populations.
        self.trace_line_Ip_minusR, = self.ax_trace.plot(
            [], [], linestyle=":", color="tab:blue", linewidth=1.8,
            label="I+(-M,t) mirror")
        self.trace_line_Im_minusR, = self.ax_trace.plot(
            [], [], linestyle=":", color="goldenrod", linewidth=1.8,
            label="I-(-M,t) mirror")
        self.ax_trace.set_xlabel("time since trace start [simulation units]", fontsize=8)
        self.ax_trace.set_ylabel("local intensity [arb.]", fontsize=8)
        self.ax_trace.legend(loc="best", ncol=2, fontsize=7)
        for ax in (self.ax_spec, self.ax_tensor, self.ax_trace):
            ax.tick_params(labelsize=7)

    def _start_new_trace(self, record_now: bool = False) -> None:
        """Reset all four histories together without changing the live model.

        Also used after material/state loading, where the parent creates a new
        canvas and replaces the model. Never retain mirrors from an old state.
        """
        self.trace_t0 = self.model.t
        self.trace_start_R = float(self.model.params.rf_burn_R)
        self.trace_t = []
        self.trace_Ip_R = []
        self.trace_Im_R = []
        self.trace_Ip_minusR = []
        self.trace_Im_minusR = []
        self.trace_start_vals = self.model.local_intensities(
            self.trace_start_R, use_reference=False)
        mirror = self.model.local_intensities(-self.trace_start_R, use_reference=False)
        self.trace_start_vals.update(
            Iplus_mirror=mirror["Iplus"], Iminus_mirror=mirror["Iminus"])
        if record_now:
            self._record_trace_point()

    def _record_trace_point(self) -> None:
        """Record absolute direct/mirror intensities at the same simulation time."""
        vals = self.model.pair_intensities(self.model.params.rf_burn_R, use_reference=False)
        self.trace_t.append(float(self.model.t - self.trace_t0))
        self.trace_Ip_R.append(vals["Iplus_R"])
        self.trace_Im_R.append(vals["Iminus_R"])
        self.trace_Ip_minusR.append(vals["Iplus_minusR"])
        self.trace_Im_minusR.append(vals["Iminus_minusR"])
        if len(self.trace_t) > self.trace_max_points:
            for history in (self.trace_t, self.trace_Ip_R, self.trace_Im_R,
                            self.trace_Ip_minusR, self.trace_Im_minusR):
                del history[:-self.trace_max_points]

    def _set_R(self, value):
        # Monitoring does not affect commands, schedules, or populations.
        k = nearest_bin(self.model.Rplus, value)
        R = float(self.model.Rplus[k])
        changed = R != self.model.params.rf_burn_R
        self.model.params.rf_burn_R = R
        with blocked(self.R_box): self.R_box.setValue(R)
        with blocked(self.monitor_bin_box): self.monitor_bin_box.setValue(k)
        if changed: self._start_new_trace(record_now=True)
        if hasattr(self, 'ax_spec'): self._update_plots()

    def _set_monitor_bin(self, value):
        self._set_R(float(self.model.Rplus[int(value)]))

    def _on_spectrum_click(self, event):
        # No accidental click-to-select. Use Marker positions or its explicit dialog.
        return

    def _enter_monitor_R(self):
        running = self.timer.isActive()
        self.timer.stop()
        try:
            d = ExactValuesDialog("Set monitor marker M", [
                ('R', 'Physical R', self.model.params.rf_burn_R,
                 self.model.params.r_min, self.model.params.r_max)], self,
                 explanation='Choose the monitoring location. R snaps to the nearest bin. '
                             'The mirror follows at -R. RF pulse positions and populations are unchanged.')
            if accepted(d, run_dialog(d)): self._set_R(d.values['R'])
        finally:
            if running: self.timer.start()

    def _refresh_profile_picker(self):
        old = self.profile_view_combo.currentData()
        with blocked(self.profile_view_combo):
            self.profile_view_combo.clear()
            self.profile_view_combo.addItem('Hide profile bounds', -1)
            for i, p in enumerate(self.model.program.profiles):
                self.profile_view_combo.addItem(p.name + ('' if p.enabled else ' [disabled]'), i)
            desired = old if old is not None else -1
            index = self.profile_view_combo.findData(desired)
            self.profile_view_combo.setCurrentIndex(max(0, index))

    def _profile_view_changed(self, *args):
        if hasattr(self, 'ax_spec'): self._update_plots()

    def _update_profile_markers(self):
        i = self.profile_view_combo.currentData()
        p = self.model.program.profiles[i] if i is not None and 0 <= i < len(self.model.program.profiles) else None
        visible = p is not None and bool(p.pulses)
        for obj in (self.profile_left_line, self.profile_right_line, self.left_text, self.right_text):
            obj.set_visible(visible)
        if not visible:
            self.profile_bounds_label.setText('No profile limits shown. Set pulse locations in the per-bin editor.')
            return
        bins = [pulse.bin_index for pulse in p.pulses]
        lo, hi = min(bins), max(bins)
        a, b = float(self.model.Rplus[lo]), float(self.model.Rplus[hi])
        self.profile_left_line.set_xdata([a, a]); self.profile_right_line.set_xdata([b, b])
        self.left_text.set_position((a, .82)); self.right_text.set_position((b, .74))
        self.profile_bounds_label.setText(f'L={a:+.6f} (bin {lo}); R={b:+.6f} (bin {hi}). '
            'These are configured bounds; gaps and disabled bins are not filled in.')

    def _set_gain(self, value):
        try:
            self.model.set_program_gain(value)
            self._update_plots()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self,"RF gain",str(exc))

    def _set_pause_at_end(self, checked):
        self.pause_at_program_end = bool(checked)

    def _update_rf_button(self):
        # This is also called while the inherited controls are being built.
        controller = PulsePlaybackController(self.model)
        self.rf_button.setText(controller.button_text(paused=getattr(self, "paused", False)))
        self.rf_button.setStyleSheet("font-weight: bold; padding: 6px;")
        self.rf_button.setToolTip(
            "Click to start/replay a loaded program, or unmute a running program. "
            "Loading alone does not apply RF. RF OFF mutes an active schedule; "
            "Pause simulation freezes its clock.")

    def _sync_rf_button(self):
        # Qt setChecked emits toggled, so synchronization must not initiate
        # another run. Only the explicit user action starts the program.
        with blocked(self.rf_button):
            self.rf_button.setChecked(bool(self.model.params.rf_enabled))
        self._update_rf_button()

    def _toggle_rf(self, checked):
        controller = PulsePlaybackController(self.model)
        try:
            if checked:
                action = controller.rf_on()
                self.paused = False
                self.pause_button.setText("Pause simulation")
                if action in ("started", "restarted"):
                    self._start_new_trace(record_now=True)
            else:
                controller.rf_off()
            self._sync_rf_button()
            self._record_trace_point()
            self._update_plots()
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            controller.rf_off()
            self._sync_rf_button()
            self._update_plots()
            QtWidgets.QMessageBox.warning(self, "RF not started", str(exc))

    def _start_program(self):
        try:
            PulsePlaybackController(self.model).restart()
            self._sync_rf_button()
            self.paused = False
            self.pause_button.setText("Pause simulation")
            self._start_new_trace(record_now=True)
            self._update_plots()
        except ValueError as exc:
            self.model.set_rf_enabled(False)
            self._sync_rf_button()
            QtWidgets.QMessageBox.warning(self, "RF program", str(exc))

    def _stop_program(self):
        self.model.stop_program()
        self._sync_rf_button()
        self._update_plots()

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Run simulation" if self.paused else "Pause simulation")
        self._update_plots()

    def _apply_program(self, program):
        self.model.set_program(program)
        self._refresh_profile_picker()
        self._sync_rf_button()
        with blocked(self.gain_box): self.gain_box.setValue(self.model.program.gain)
        self._update_plots()

    def _edit_program(self):
        # Editing must not consume a pulse while the modal dialog is open.
        was_paused = self.paused
        self.paused = True
        timer_running = self.timer.isActive()
        self.timer.stop()
        dialog = ProfileEditor(self.model.program,self.model.params.rf_burn_R,self)
        try:
            result = dialog.exec() if hasattr(dialog,"exec") else dialog.exec_()
            accepted = QtWidgets.QDialog.DialogCode.Accepted if hasattr(QtWidgets.QDialog,"DialogCode") else QtWidgets.QDialog.Accepted
            if result == accepted:
                self._apply_program(dialog.program)
        finally:
            self.paused = was_paused
            if timer_running: self.timer.start()
            self._update_plots()

    def _load_program(self):
        # Freeze the simulation while interacting with the file dialog.
        running = self.timer.isActive(); self.timer.stop()
        try:
            path,_ = QtWidgets.QFileDialog.getOpenFileName(self,"Load ideal RF program","","RF programs (*.json *.csv)")
            if path:
                p = PulseProgram.load_csv(path) if Path(path).suffix.lower()==".csv" else PulseProgram.load(path)
                self._apply_program(p)
        except (ValueError,OSError) as exc:
            QtWidgets.QMessageBox.warning(self,"RF program",str(exc))
        finally:
            if running: self.timer.start()

    def _save_program(self):
        running = self.timer.isActive(); self.timer.stop()
        try:
            path,_ = QtWidgets.QFileDialog.getSaveFileName(self,"Save ideal RF program","rf_program.json","JSON (*.json)")
            if path: self.model.program.save(path)
        except (ValueError,OSError) as exc:
            QtWidgets.QMessageBox.warning(self,"RF program",str(exc))
        finally:
            if running: self.timer.start()

    def _export_results(self):
        running = self.timer.isActive(); self.timer.stop()
        try:
            path,_ = QtWidgets.QFileDialog.getSaveFileName(self,"Export displayed monitor history","monitor_trace.csv","CSV (*.csv)")
            if not path: return
            target=Path(path)
            with target.open("w",newline="",encoding="utf-8") as f:
                writer=csv.writer(f)
                # Retain the original first five CSV columns and append mirrors.
                writer.writerow(["time_since_selection", "simulation_time", "monitor_R",
                                 "Iplus", "Iminus", "mirror_R", "Iplus_mirror", "Iminus_mirror"])
                writer.writerows((t, self.trace_t0+t, self.trace_start_R, a, b,
                                  -self.trace_start_R, am, bm)
                                 for t, a, b, am, bm in zip(
                                     self.trace_t, self.trace_Ip_R, self.trace_Im_R,
                                     self.trace_Ip_minusR, self.trace_Im_minusR))
            exposure_path=target.with_name(target.stem+"_exposure.csv")
            with exposure_path.open("w",newline="",encoding="utf-8") as f:
                writer=csv.writer(f)
                writer.writerow(["bin_index","R","delivered_base_exposure_last_run","delivered_base_exposure_since_reset"])
                writer.writerows(zip(range(len(self.model.Rplus)),self.model.Rplus,
                                     self.model.delivered_exposure,self.model.total_delivered_exposure))
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self,"Export",str(exc))
        finally:
            if running: self.timer.start()

    def _reset_model(self):
        self.model.stop_program()
        self.model.reset()
        self.rf_button.setChecked(False)
        self.model.set_rf_enabled(False)
        self.model.set_dnp_enabled(self.dnp_is_on())
        self._start_new_trace(record_now=True)
        self._update_plots()

    def _tick(self):
        try:
            if not self.paused:
                self.model.set_rf_enabled(self.rf_is_on())
                self.model.set_dnp_enabled(self.dnp_is_on())
                update = PulsePlaybackController(self.model).advance(
                    n_steps=self.steps_per_tick,
                    pause_at_end=bool(getattr(self, "pause_at_program_end", False)))
                if update.pause_requested:
                    self.paused = True
                    self.pause_button.setText("Run simulation (continue recovery)")
                if update.completed:
                    self._sync_rf_button()
                self._record_trace_point()
            self._update_plots()
        except (ValueError,FloatingPointError,RuntimeError) as exc:
            self.paused = True
            self.pause_button.setText("Run simulation")
            self.model.stop_program()
            self._sync_rf_button()
            QtWidgets.QMessageBox.critical(self,"Simulation stopped safely",str(exc))

    def _update_plots(self):
        # One spectrum snapshot per refresh. Reuse its two branch arrays for
        # both the upper sum and the middle subtraction, including the SAME
        # realization of display-only noise. No new convolution/normalization.
        R,Ip,Im,total=self.model.spectrum(noise_sigma=self.model.params.noise_sigma)
        Rb=self.model.params.rf_burn_R
        vals=self.model.response_values(Rb)
        self.line_Ip.set_data(R,Ip); self.line_Im.set_data(R,Im); self.line_total.set_data(R,total)
        tensor = Ip - Im
        self.line_tensor.set_data(R, tensor)
        self.tensor_monitor_line.set_xdata([Rb, Rb])
        self.tensor_mirror_line.set_xdata([-Rb, -Rb])
        finite_tensor = self._finite_values(tensor)
        if finite_tensor.size:
            low = min(0.0, float(finite_tensor.min()))
            high = max(0.0, float(finite_tensor.max()))
            pad = max(1e-8, 0.08 * (high-low))
            self.ax_tensor.set_ylim(low-pad, high+pad)
        preview=self.model.preview_rf_field(); active=self.model.applied_rf_field()
        self.line_preview.set_data(R,preview); self.line_rf_profile.set_data(R,active)
        self.ax_rf_profile.set_ylim(0, max(0.1,1.1*float(preview.max()),1.1*float(active.max())))
        self.burn_line.set_xdata([Rb,Rb]); self.mirror_line.set_xdata([-Rb,-Rb])
        self.mirror_R_readout.setText(f"{-Rb:+.9f}")
        self.monitor_text.set_position((Rb, .99)); self.mirror_text.set_position((-Rb, .91))
        self.monitor_text.set_text(f"M {Rb:+.3f}"); self.mirror_text.set_text(f"-M {-Rb:+.3f}")
        self._update_profile_markers()
        self._set_marker(self.point_Ip,Rb,vals['Iplus_R']); self._set_marker(self.point_Im,Rb,vals['Iminus_R'])
        self._set_marker(self.point_Ip_m,-Rb,vals['Iplus_minusR']); self._set_marker(self.point_Im_m,-Rb,vals['Iminus_minusR'])
        self.ax_spec.relim(); self.ax_spec.autoscale_view()
        self.ax_spec.set_xlim(self.model.Rplus[0]-0.03,self.model.Rplus[-1]+0.03)
        self.trace_line_Ip_R.set_data(self.trace_t,self.trace_Ip_R)
        self.trace_line_Im_R.set_data(self.trace_t,self.trace_Im_R)
        self.trace_line_Ip_minusR.set_data(self.trace_t,self.trace_Ip_minusR)
        self.trace_line_Im_minusR.set_data(self.trace_t,self.trace_Im_minusR)
        y=self._finite_values(self.trace_Ip_R, self.trace_Im_R,
                              self.trace_Ip_minusR, self.trace_Im_minusR)
        if y.size:
            lo,hi=float(y.min()),float(y.max())
            pad=max(1e-8,0.12*max(hi-lo,0.03*max(abs(hi),abs(lo),1e-12)))
            self.ax_trace.set_ylim(lo-pad,hi+pad)
        self.ax_trace.set_xlim(0,max(0.5,self.trace_t[-1]+0.05 if self.trace_t else 0.5))
        self.ax_trace.set_title(
            f"Monitor M={Rb:+.4f}; dotted mirrors at -M={-Rb:+.4f}", fontsize=9)
        self._update_info_label()
        self.canvas.draw_idle()

    def _update_info_label(self):
        pol=self.model.polarizations()
        self.p_readout.setText(f"P(t) = {pol['P']:+.5f}   ({100*pol['P']:+.2f}%)")
        self.q_readout.setText(f"Q(t)={pol['Q']:+.5f}     Q_B[P]={pol['Q_boltz_at_P']:+.5f}")
        self._sync_rf_button()
        playback = PulsePlaybackController(self.model)
        status = playback.status_text(paused=self.paused)
        self.program_label.setText(status)
        self.statusBar().showMessage(status.replace("\n", "  |  "))
        rates=self.model.effective_local_rates()
        diff=self.model.local_diffusion_diagnostics(dnp_on=self.dnp_is_on())
        eq=self.model.recovery_equilibrium_diagnostics()
        gate_factor=1 if self.rf_is_on() else 0
        self.info_label.setText(
            f"Simulation t={self.model.t:.5f}\n"
            f"Monitor R={self.model.params.rf_burn_R:.6f}\n"
            f"Applied base rate at monitor={gate_factor*rates['command_at_R']:.5g}\n"
            f"Effective Γ+={gate_factor*rates['gamma_rf_Iplus_R']:.5g}; "
            f"Γ-={gate_factor*rates['gamma_rf_Iminus_R']:.5g}\n"
            f"Diffusion dI+/dt={diff['dIplus_diff_dt']:+.4e}\n"
            f"Diffusion dI-/dt={diff['dIminus_diff_dt']:+.4e}\n"
            f"Q-Q_B(P)={eq['Q_minus_Q_B']:+.3e}\n"
            f"Boltzmann population RMS={eq['fraction_rms_error']:.3e}\n"
            + ("\n".join(eq['warnings'])+"\n" if eq['warnings'] else "") +
            "RF power is a base-rate setting, not watts. Existing packet coupling "
            "weights still apply; no Voigt spreading is used."
        )


def main(argv=None):
    app=QtWidgets.QApplication(sys.argv if argv is None else argv)
    win=Spin1RealtimeWindow()
    screen=app.primaryScreen()
    if screen is not None:
        geom=screen.availableGeometry()
        win.resize(min(1000,int(0.95*geom.width())), min(555,int(0.88*geom.height())))
    win.show()
    return int(app.exec() if hasattr(app,"exec") else app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
