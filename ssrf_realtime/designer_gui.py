"""Material persistence and snapshot-based synchronized tensor designer.

The accepted two-panel live GUI and manual pulse editor remain in gui.py.
This subclass adds workflow controls without changing population dynamics.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import argparse
import copy
import json
import sys
import threading
import numpy as np

from .gui import Spin1RealtimeWindow as ExistingWindow
from .baseline_gui import QtCore, QtWidgets, FigureCanvas, Figure
from .safe_widgets import ArrowDoubleSpinBox, ArrowSpinBox, NoWheelComboBox
from .numeric_dialogs import accepted, run_dialog
from .material_config import MaterialConfig, PopulationSnapshot
from .tensor_optimizer import OptimizerSettings, optimize_tensor, OptimizationCancelled

Signal=getattr(QtCore,'pyqtSignal',None) or QtCore.Signal


def label(text):
    w=QtWidgets.QLabel(text); w.setWordWrap(True); return w


class JSONEditor(QtWidgets.QDialog):
    """All fields, including backend-only parameters, through explicit confirmed entry."""
    def __init__(self,title,data,validator,parent=None):
        super().__init__(parent); self.setWindowTitle(title); self.value=None; self.validator=validator
        layout=QtWidgets.QVBoxLayout(self)
        layout.addWidget(label('Edit exact values here, then validate and apply. Live numeric fields remain '
                               'arrow-only. Unknown keys, nonfinite values and incompatible settings are rejected.'))
        self.edit=QtWidgets.QPlainTextEdit(json.dumps(data,indent=2,allow_nan=False))
        self.edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap if hasattr(QtWidgets.QPlainTextEdit,'LineWrapMode') else QtWidgets.QPlainTextEdit.NoWrap)
        layout.addWidget(self.edit)
        self.error=label(''); layout.addWidget(self.error)
        row=QtWidgets.QHBoxLayout()
        b=QtWidgets.QPushButton('Cancel'); b.clicked.connect(self.reject); row.addWidget(b)
        b=QtWidgets.QPushButton('Validate and apply'); b.clicked.connect(self.commit); row.addWidget(b)
        layout.addLayout(row); self.resize(700,500)
    def commit(self):
        try:
            data=json.loads(self.edit.toPlainText())
            self.value=self.validator(data)
        except (ValueError,TypeError,KeyError) as exc:
            self.error.setText(str(exc)); return
        self.accept()


class ApplyMaterialDialog(QtWidgets.QDialog):
    def __init__(self,name,parent=None):
        super().__init__(parent); self.setWindowTitle('Apply material configuration')
        layout=QtWidgets.QVBoxLayout(self)
        layout.addWidget(label(f'Apply material: {name}\nThe RF schedule will stop and the simulation will remain paused.'))
        self.preserve=QtWidgets.QRadioButton('Keep the current populations and simulation time')
        self.reset=QtWidgets.QRadioButton('Reinitialize explicitly from the file\'s initial P / Q and grid')
        self.preserve.setChecked(True)
        layout.addWidget(self.preserve); layout.addWidget(self.reset)
        layout.addWidget(label('Keeping populations requires the same grid and static lineshape/capacities. '
                               'Changing p0 or q0 in this mode affects the next explicit reset only. '
                               'Material files contain no RF pulse table and no manipulated state.'))
        row=QtWidgets.QHBoxLayout()
        for text,fun in [('Cancel',self.reject),('Apply material',self.accept)]:
            b=QtWidgets.QPushButton(text); b.clicked.connect(fun); row.addWidget(b)
        layout.addLayout(row); self.resize(490,260)


class DesignerWorker(QtCore.QThread):
    message=Signal(str)
    completed=Signal(object)
    failed=Signal(str)
    def __init__(self,snapshot,settings,parent=None):
        super().__init__(parent); self.snapshot=snapshot; self.settings=settings; self.cancel_event=threading.Event()
    def run(self):
        try:
            result=optimize_tensor(self.snapshot,self.settings,self.message.emit,self.cancel_event)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class TensorDesignerDialog(QtWidgets.QDialog):
    def __init__(self,snapshot,settings,material,parent=None):
        super().__init__(parent)
        self.setWindowTitle('Synchronized tensor-profile designer — current state')
        self.snapshot=snapshot; self.settings=OptimizerSettings.from_dict(asdict(settings))
        self.material=material; self.result=None; self.install_requested=False; self.worker=None
        self.controls={}
        layout=QtWidgets.QVBoxLayout(self)
        layout.addWidget(label('All selected bins start together and stop at one common time T. '
            'Each bin has its own constant power U. Objective: positive Q immediately at T. '
            'The live state is paused and unchanged. This searches a bounded pulse family, not a certified global optimum.'))
        p=snapshot.parameters
        self.flags=label(f"Captured simulation t={snapshot.simulation_time:.6g}; "
            f"DNP {'ON' if p['dnp_enabled'] else 'OFF'}, diffusion {'ON' if p['diffusion_enabled'] else 'OFF'}, "
            f"T1 rate={p['t1_rate']:g}. All settings are used in the prediction. U is NOT watts.")
        layout.addWidget(self.flags)
        form=QtWidgets.QGridLayout()
        specs=[('max_power','Maximum RF command per bin',.001,1e6,1.,6),
               ('min_duration','Minimum common duration',1e-6,1e5,.01,6),
               ('max_duration','Maximum common duration',1e-6,1e5,.1,6),
               ('duration_samples','Duration samples',2,40,1,None),
               ('max_iterations','Power iterations per start',1,1000,5,None),
               ('max_wall_seconds','Search budget [wall seconds]',1,86400,30.,1)]
        for i,(key,text,lo,hi,step,dec) in enumerate(specs):
            box=ArrowSpinBox() if dec is None else ArrowDoubleSpinBox()
            if dec is not None: box.setDecimals(dec)
            box.setRange(lo,hi); box.setSingleStep(step); box.setValue(getattr(self.settings,key))
            self.controls[key]=box
            box.valueChanged.connect(self._invalidate)
            form.addWidget(QtWidgets.QLabel(text),i//2,2*(i%2)); form.addWidget(box,i//2,2*(i%2)+1)
        layout.addLayout(form)
        self.candidate_combo=NoWheelComboBox()
        for title,mode in [('Negative subtraction OR positive RF slope','union'),
                           ('Negative signed I+ - I- only','tensor_deficit'),
                           ('Positive initial RF tensor slope only','rf_gain'),('All bins (bounded full-line search)','all')]:
            self.candidate_combo.addItem(title,mode)
        self.candidate_combo.setCurrentIndex(self.candidate_combo.findData(self.settings.candidate_mode))
        self.candidate_combo.currentIndexChanged.connect(self._invalidate)
        row=QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel('Candidate bins')); row.addWidget(self.candidate_combo)
        self.advanced_button=QtWidgets.QPushButton('Exact / advanced settings...')
        self.advanced_button.clicked.connect(self._advanced); row.addWidget(self.advanced_button); layout.addLayout(row)
        layout.addWidget(label('The signed subtraction is used for either sign of P. Full rate equations decide the endpoint. '
            'Advanced JSON includes candidate regions, individual power ceilings, multistarts, tolerances and numerical limits. '
            'The default ceiling 20 and time range 0.01–2 are trial search limits, not calibrated hardware values.'))
        actions=QtWidgets.QHBoxLayout()
        self.generate_button=QtWidgets.QPushButton('Calculate from captured state')
        self.generate_button.clicked.connect(self._generate)
        self.cancel_button=QtWidgets.QPushButton('Cancel calculation'); self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        self.export_button=QtWidgets.QPushButton('Export JSON / CSV + snapshot...'); self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        for b in [self.generate_button,self.cancel_button,self.export_button]: actions.addWidget(b)
        layout.addLayout(actions)
        self.tabs=QtWidgets.QTabWidget();self.tabs.setMinimumHeight(140)
        self.log=QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        self.tabs.addTab(self.log,'Progress / validation')
        self.fig=Figure(figsize=(7,3),constrained_layout=True); self.canvas=FigureCanvas(self.fig)
        self.tabs.addTab(self.canvas,'Predicted endpoint / RF profile')
        self.table=QtWidgets.QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(['Bin','Physical R','Power U','Start','Duration'])
        triggers=QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers if hasattr(QtWidgets.QAbstractItemView,'EditTrigger') else QtWidgets.QAbstractItemView.NoEditTriggers
        self.table.setEditTriggers(triggers)
        self.tabs.addTab(self.table,'Generated bin commands')
        layout.addWidget(self.tabs,1)
        self.summary=label('No calculation yet. Generating never applies RF. Load the result, then click RF ON.'); layout.addWidget(self.summary)
        actions=QtWidgets.QHBoxLayout()
        self.install_button=QtWidgets.QPushButton('Load generated program (do not run)'); self.install_button.setEnabled(False)
        self.install_button.clicked.connect(self._install)
        self.close_button=QtWidgets.QPushButton('Close — leave simulation paused'); self.close_button.clicked.connect(self.reject)
        actions.addWidget(self.install_button); actions.addWidget(self.close_button); layout.addLayout(actions)
        self.resize(900,640)
        screen=QtWidgets.QApplication.primaryScreen()
        if screen: self.resize(min(900,int(screen.availableGeometry().width()*.95)),min(640,int(screen.availableGeometry().height()*.90)))

    def _invalidate(self,*_):
        self.result=None
        if hasattr(self,'export_button'): self.export_button.setEnabled(False); self.install_button.setEnabled(False)
        if hasattr(self,'summary'): self.summary.setText('Settings changed; recalculate before export or installation.')

    def _read_settings(self):
        data=asdict(self.settings)
        for k,b in self.controls.items(): data[k]=b.value()
        data['candidate_mode']=self.candidate_combo.currentData()
        self.settings=OptimizerSettings.from_dict(data)
        return self.settings

    def _advanced(self):
        try: self._read_settings()
        except ValueError as exc: QtWidgets.QMessageBox.warning(self,'Settings',str(exc)); return
        dialog=JSONEditor('Exact tensor-search settings',asdict(self.settings),OptimizerSettings.from_dict,self)
        if accepted(dialog,run_dialog(dialog)):
            self.settings=dialog.value
            for k,b in self.controls.items():
                value=getattr(self.settings,k)
                b.blockSignals(True);b.setRange(min(b.minimum(),value),max(b.maximum(),value));b.setValue(value);b.blockSignals(False)
            self.candidate_combo.setCurrentIndex(self.candidate_combo.findData(self.settings.candidate_mode))
            self._invalidate()

    def _set_busy(self,busy):
        for w in list(self.controls.values())+[self.candidate_combo,self.advanced_button,self.generate_button,self.close_button]: w.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        if busy: self.export_button.setEnabled(False); self.install_button.setEnabled(False)

    def _generate(self):
        try: self._read_settings()
        except ValueError as exc: QtWidgets.QMessageBox.warning(self,'Settings',str(exc)); return
        self.result=None; self.log.clear(); self._set_busy(True)
        self.worker=DesignerWorker(self.snapshot,OptimizerSettings.from_dict(asdict(self.settings)),self)
        self.worker.message.connect(self.log.appendPlainText)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(lambda:self._set_busy(False))
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel_event.set(); self.cancel_button.setEnabled(False)
            self.log.appendPlainText('Cancellation requested; no live populations have changed.')

    def _failed(self,text):
        self.summary.setText(text); self.log.appendPlainText(text)

    def _completed(self,result):
        self.result=result
        r=result.report
        self.summary.setText(f"{r['status']}: P {r['initial']['P']:+.6f} → {r['final']['P']:+.6f}; "
            f"Q {r['initial']['Q']:+.6f} → {r['final']['Q']:+.6f}; T={result.duration:.6g}; "
            f"{r['active_bins']} active bins; playback dt={r['recommended_playback_dt']:.6g}. RF advantage over no-RF endpoint: {r['RF_advantage_over_no_RF']:+.6g}.")
        self.log.appendPlainText(json.dumps(r,indent=2,allow_nan=False))
        self.export_button.setEnabled(True)
        self.install_button.setEnabled(r['status']=='completed' and r['active_bins']>0)
        self.table.setRowCount(r['active_bins'])
        for row,j in enumerate(np.flatnonzero(result.powers>0)):
            for col,v in enumerate([str(j),f'{result.snapshot.grid[j]:+.9g}',f'{result.powers[j]:.9g}','0',f'{result.duration:.9g}']):
                self.table.setItem(row,col,QtWidgets.QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        self.fig.clear(); ax=self.fig.add_subplot(111); model=result.snapshot.make_model()
        R,ap,am,total=model.spectrum_from_state(result.snapshot.n)
        _,bp,bm,after=model.spectrum_from_state(result.final_n)
        ax.plot(R,ap,linestyle=':',label='Initial I+')
        ax.plot(R,am,linestyle=':',label='Initial I-')
        ax.plot(R,bp,label='Endpoint I+'); ax.plot(R,bm,label='Endpoint I-')
        ax.set_xlabel('physical R'); ax.set_ylabel('intensity [display units]')
        ax2=ax.twinx(); ax2.step(R,result.powers,where='mid',linestyle='--',alpha=.5,label='RF command U')
        ax2.set_ylabel('command U'); ax2.set_ylim(0,max(.1,result.settings.max_power*1.08))
        h,l=ax.get_legend_handles_labels();h2,l2=ax2.get_legend_handles_labels()
        ax.legend(h+h2,l+l2,fontsize=8,ncol=3)
        self.canvas.draw_idle()

    def _export(self):
        if self.result is None:return
        directory=QtWidgets.QFileDialog.getExistingDirectory(self,'Select parent directory for a new design bundle')
        if not directory:return
        root=Path(directory)/'tensor_design'
        i=1
        while root.exists(): root=Path(directory)/f'tensor_design_{i:02d}'; i+=1
        try:
            self.result.export(root,self.material)
            self.log.appendPlainText(f'Exported {root} (rf_program.json and rf_program.csv plus reproducibility files).')
        except (OSError,ValueError) as exc: QtWidgets.QMessageBox.warning(self,'Export',str(exc))

    def _install(self):
        if self.result is None or self.result.report['status']!='completed':return
        self.install_requested=True; self.accept()

    def reject(self):
        if self.worker and self.worker.isRunning(): self._cancel(); return
        super().reject()

    def closeEvent(self,event):
        if self.worker and self.worker.isRunning(): self._cancel(); event.ignore()
        else: event.accept()


class Spin1RealtimeWindow(ExistingWindow):
    def __init__(self,params=None):
        self.optimizer_settings=OptimizerSettings()
        self.material_name='Uncalibrated demonstration'
        self.material_description='Editable demonstration values; not a calibrated material.'
        self.material_path=None
        super().__init__(params)
        self.setWindowTitle('Real-time ss-RF — tensor designer / RF playback fix')

    def _spin_box(self,label_text,value,lo,hi,step,decimals,callback):
        # Values loaded from a config must not be visually clipped to an old GUI range.
        return super()._spin_box(label_text,value,min(lo,value),max(hi,value),step,
                                max(decimals, min(10,len(str(value).split('.')[-1]))),callback)

    def _build_ui(self):
        super()._build_ui()
        program_group=self.edit_button.parentWidget()
        side=program_group.parentWidget().layout()
        box=QtWidgets.QGroupBox('Material configuration / tensor design')
        layout=QtWidgets.QVBoxLayout(box)
        self.material_label=label(self.material_name); layout.addWidget(self.material_label)
        row=QtWidgets.QHBoxLayout()
        for text,fun in [('Load material',self._load_material),('Save material',self._save_material),('Save as...',self._save_material_as)]:
            b=QtWidgets.QPushButton(text); b.clicked.connect(fun);row.addWidget(b)
        layout.addLayout(row)
        self.all_settings_button=QtWidgets.QPushButton('Edit ALL material / search settings...')
        self.all_settings_button.clicked.connect(self._edit_all_settings); layout.addWidget(self.all_settings_button)
        row=QtWidgets.QHBoxLayout()
        for text,fun in [('Save state',self._save_state),('Load state',self._load_state)]:
            b=QtWidgets.QPushButton(text);b.clicked.connect(fun);row.addWidget(b)
        layout.addLayout(row)
        self.tensor_button=QtWidgets.QPushButton('Generate tensor program from CURRENT state...')
        self.tensor_button.setToolTip('Pause and capture the live populations, calculate synchronized per-bin powers, preview and export JSON/CSV. No automatic RF.')
        self.tensor_button.clicked.connect(self._design_tensor);layout.addWidget(self.tensor_button)
        side.insertWidget(0,box)

    @contextmanager
    def _frozen(self,leave_paused=False):
        running=self.timer.isActive(); was_paused=self.paused; self.timer.stop(); self.paused=True
        try: yield
        finally:
            self.paused=True if leave_paused else was_paused
            self.pause_button.setText('Run simulation' if self.paused else 'Pause simulation')
            if running:self.timer.start()
            self._update_plots()

    def _capture_material(self):
        gui={'steps_per_tick':int(self.steps_per_tick),'trace_max_points':int(self.trace_max_points),
             'timer_interval_ms':int(self.timer.interval()),'window_width':max(500,int(self.width())),
             'window_height':max(350,int(self.height()))}
        return MaterialConfig.capture(self.model,self.material_name,self.optimizer_settings,gui,self.material_description)

    def _adopt_model(self,model,config=None):
        # Complete candidate exists before replacing anything. Never reset a live
        # state piecemeal through connected widget callbacks.
        self.model=model; self.model.stop_program()
        j=int(np.argmin(abs(model.Rplus-model.params.rf_burn_R)))
        model.params.rf_burn_R=float(model.Rplus[j])
        if config:
            self.material_name=config.name;self.material_description=config.description
            self.optimizer_settings=OptimizerSettings.from_dict(config.optimizer)
        self.trace_t0=model.t; self.trace_t=[]; self.trace_Ip_R=[]; self.trace_Im_R=[]
        self.trace_start_R=float(model.params.rf_burn_R);self.trace_start_vals={}
        old=self.takeCentralWidget()
        self._build_ui(); self._init_plots()
        if old is not None: old.deleteLater()
        if config:
            self.steps_box.blockSignals(True);self.steps_box.setValue(config.gui['steps_per_tick']);self.steps_box.blockSignals(False)
            self.steps_per_tick=config.gui['steps_per_tick']; self.trace_max_points=config.gui['trace_max_points']
            self.timer.setInterval(config.gui['timer_interval_ms'])
            screen=QtWidgets.QApplication.primaryScreen()
            w,h=config.gui['window_width'],config.gui['window_height']
            if screen:w=min(w,int(screen.availableGeometry().width()*.95));h=min(h,int(screen.availableGeometry().height()*.9))
            self.resize(w,h)
        self.paused=True;self.pause_button.setText('Run simulation')
        self._start_new_trace(record_now=True);self._update_plots()

    def _apply_material(self,config):
        d=ApplyMaterialDialog(config.name,self)
        if not accepted(d,run_dialog(d)):return False
        model=config.make_model(self.model,preserve_state=d.preserve.isChecked())
        self._adopt_model(model,config)
        return True

    def _load_material(self):
        with self._frozen(leave_paused=True):
            path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Load material configuration','','Material JSON (*.json)')
            if not path:return
            try:
                config=MaterialConfig.load(path)
                if self._apply_material(config):self.material_path=Path(path)
            except (ValueError,OSError,TypeError) as exc:QtWidgets.QMessageBox.warning(self,'Material not loaded',str(exc))

    def _save_material_as(self): self._save_material(force_as=True)
    def _save_material(self,checked=False,force_as=False):
        with self._frozen():
            path=self.material_path
            if force_as or path is None:
                text,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save material configuration',str(path or 'material.json'),'Material JSON (*.json)')
                if not text:return
                path=Path(text)
                if path.suffix.lower()!='.json':path=path.with_suffix('.json')
            try:
                self._capture_material().save(path);self.material_path=path
                self.material_label.setText(f'{self.material_name}\n{path.name}')
            except (ValueError,OSError) as exc:QtWidgets.QMessageBox.warning(self,'Material save',str(exc))

    def _edit_all_settings(self):
        with self._frozen(leave_paused=True):
            d=JSONEditor('All material and tensor-search parameters',self._capture_material().to_dict(),MaterialConfig.from_dict,self)
            if accepted(d,run_dialog(d)):
                try:self._apply_material(d.value)
                except (ValueError,OSError) as exc:QtWidgets.QMessageBox.warning(self,'Settings not applied',str(exc))

    def _save_state(self):
        with self._frozen():
            path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save current population snapshot','state.npz','Population snapshot (*.npz)')
            if path:
                try:PopulationSnapshot.capture(self.model).save(path)
                except (ValueError,OSError) as exc:QtWidgets.QMessageBox.warning(self,'Snapshot',str(exc))

    def _load_state(self):
        with self._frozen(leave_paused=True):
            path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Restore population snapshot (replaces current state)','','Population snapshot (*.npz)')
            if not path:return
            buttons=QtWidgets.QMessageBox.StandardButton if hasattr(QtWidgets.QMessageBox,'StandardButton') else QtWidgets.QMessageBox
            if QtWidgets.QMessageBox.question(self,'Replace live state?',
                'Restore this snapshot\'s populations, time and material parameters? RF remains off and the simulation paused.',
                buttons.Yes|buttons.No,buttons.No)!=buttons.Yes:return
            try:
                snapshot=PopulationSnapshot.load(path)
                model=snapshot.make_model()
                # Keep a compatible planned program; it is not started.
                try:model.set_program(self.model.program)
                except ValueError:pass
                config=MaterialConfig.capture(model,'Snapshot: '+Path(path).stem,self.optimizer_settings,
                                               self._capture_material().gui,'Parameters restored from population snapshot.')
                self._adopt_model(model,config);self.material_path=None
            except (ValueError,OSError,KeyError) as exc:QtWidgets.QMessageBox.warning(self,'Snapshot not loaded',str(exc))

    def _design_tensor(self):
        with self._frozen(leave_paused=True):
            snapshot=PopulationSnapshot.capture(self.model)
            d=TensorDesignerDialog(snapshot,self.optimizer_settings,self._capture_material(),self)
            run_dialog(d)
            self.optimizer_settings=d.settings
            if d.install_requested and d.result is not None:
                if not snapshot.matches(self.model):
                    QtWidgets.QMessageBox.warning(self,'Stale design','The live population state or material changed. Regenerate the program.');return
                dt=d.result.report['recommended_playback_dt']
                if dt < self.model.params.dt:
                    self.model.params.dt=dt
                    self.dt_box.blockSignals(True)
                    self.dt_box.setDecimals(max(8,self.dt_box.decimals()))
                    self.dt_box.setMinimum(min(self.dt_box.minimum(),dt));self.dt_box.setValue(dt)
                    self.dt_box.blockSignals(False)
                self._apply_program(d.result.program)
                self.pause_at_end_box.setChecked(True)
                self.material_label.setText(
                    f'{self.material_name}\nProgram READY; press RF ON to apply. '
                    'Pauses at the endpoint so you can inspect the enhancement.')


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(description='Ideal-bin ss-RF simulation with material configs and synchronized tensor design')
    parser.add_argument('--material',help='Load material JSON and initialize from it (RF off, paused)')
    parser.add_argument('--state',help='Restore snapshot NPZ instead of initialization (RF off, paused)')
    opts,qt_args=parser.parse_known_args(args)
    app=QtWidgets.QApplication([sys.argv[0]]+qt_args)
    win=Spin1RealtimeWindow()
    if opts.material or opts.state:
        with win._frozen(leave_paused=True):
            config=MaterialConfig.load(opts.material) if opts.material else None
            if opts.state:
                snapshot=PopulationSnapshot.load(opts.state);model=snapshot.make_model()
                if config and config.parameters!={**snapshot.parameters}:
                    raise ValueError('Do not combine a different material with a saved state; load it explicitly after restoring.')
            else:model=config.make_model(preserve_state=False)
            win._adopt_model(model,config)
            if config:win.material_path=Path(opts.material)
    screen=app.primaryScreen()
    if screen:
        win.resize(min(win.width(),int(.95*screen.availableGeometry().width())),
                   min(win.height(),int(.9*screen.availableGeometry().height())))
    win.show()
    return int(app.exec() if hasattr(app,'exec') else app.exec_())
