"""Execute the shipped plotting/history callbacks with real Matplotlib Agg axes.

Only Qt widgets and window construction are replaced by small doubles. No
spectrum, trace equation, plot method, or RF/playback logic is reimplemented.
This is not a native Qt rendering/mouse test.
"""
from __future__ import annotations
import ast
import csv
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams
from ssrf_realtime.profile_editing import nearest_bin

ROOT = Path(__file__).resolve().parents[1]

class Widget:
    def __init__(self, value=None):
        self.value = value
        self.text = ''
        self.blocked = False
    def blockSignals(self, value):
        old = self.blocked
        self.blocked = value
        return old
    def setValue(self, value): self.value = value
    def currentData(self): return self.value
    def setText(self, text): self.text = text

class Timer:
    def __init__(self): self.active = True
    def isActive(self): return self.active
    def stop(self): self.active = False
    def start(self): self.active = True

class FileDialog:
    next_path = ''
    @classmethod
    def getSaveFileName(cls, *args): return str(cls.next_path), 'CSV (*.csv)'

@contextmanager
def blocked(widget):
    old = widget.blockSignals(True)
    try: yield
    finally: widget.blockSignals(old)

def extract_methods(path, names):
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == 'Spin1RealtimeWindow')
    result = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in result} != set(names):
        raise RuntimeError('Not all requested GUI methods were found in source')
    return result

names = ['_init_plots', '_start_new_trace', '_record_trace_point', '_update_plots',
         '_set_R', '_set_monitor_bin', '_on_spectrum_click', '_update_profile_markers',
         '_export_results']
methods = extract_methods(ROOT/'ssrf_realtime/gui.py', names)
methods += extract_methods(ROOT/'ssrf_realtime/baseline_gui.py', ['_finite_values', '_set_marker'])
module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
    name='ActualPlotCallbacks', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[]))
MESSAGES = []
namespace = dict(np=np, csv=csv, Path=Path, nearest_bin=nearest_bin, blocked=blocked,
                 QtWidgets=SimpleNamespace(QFileDialog=FileDialog, QMessageBox=SimpleNamespace(
                     warning=lambda *args: MESSAGES.append(args[1:]))))
exec(compile(module, '<shipped GUI plotting methods>', 'exec'), namespace)
ActualPlotCallbacks = namespace['ActualPlotCallbacks']

class PlotHarness(ActualPlotCallbacks):
    def __init__(self, params=None, size=(7.2, 5.0)):
        self.model = IdealBinModel(params or IdealBinParams(n_bins=101))
        self.model.params.rf_burn_R = float(self.model.Rplus[nearest_bin(
            self.model.Rplus, self.model.params.rf_burn_R)])
        self.fig = Figure(figsize=size, constrained_layout=True, dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        # Qt normally schedules/coalesces paints. Explicit draw() still renders
        # the actual artists; avoid doing a full Agg paint on every test refresh.
        self.canvas.draw_idle = lambda: None
        self.R_box = Widget()
        self.monitor_bin_box = Widget()
        self.mirror_R_readout = Widget()
        self.profile_view_combo = Widget(-1)
        self.profile_bounds_label = Widget()
        self.timer = Timer()
        self.trace_max_points = 4500
        self.info_refreshes = 0
        self._init_plots()
        self._start_new_trace(record_now=True)
        self._update_plots()
    def _update_info_label(self): self.info_refreshes += 1
