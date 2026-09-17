"""Explicit per-bin power/timing editor with arrow-safe numeric controls.

The editor owns a copy of the program. No state, command, or clock is changed
until the main window receives Apply program. No rate equations are modified.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .baseline_gui import QtCore, QtWidgets
from .pulse_program import PulseProgram, RFProfile, BinPulse, make_profile, example_program
from .safe_widgets import ArrowDoubleSpinBox, ArrowSpinBox, NoWheelComboBox
from .profile_editing import (nearest_bin, region_indices, ensure_region_rows,
                              update_rows, parse_custom_table, custom_table_text)
from .numeric_dialogs import ExactValuesDialog, accepted, run_dialog

QT = QtCore.Qt
FLAGS = QT.ItemFlag if hasattr(QT, 'ItemFlag') else QT
CHECK = QT.CheckState if hasattr(QT, 'CheckState') else QT
VIEW = QtWidgets.QAbstractItemView
SELECTION = VIEW.SelectionBehavior if hasattr(VIEW, 'SelectionBehavior') else VIEW
EDIT = VIEW.EditTrigger if hasattr(VIEW, 'EditTrigger') else VIEW


@contextmanager
def blocked(obj):
    old = obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(old)


def number_box(value=0.0, lo=0.0, hi=1e6, decimals=9, step=0.1):
    box = ArrowDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(lo, max(hi, float(value)))
    box.setSingleStep(step)
    box.setValue(value)
    return box


class PasteTableDialog(QtWidgets.QDialog):
    def __init__(self, grid, parent=None):
        super().__init__(parent)
        self.grid = grid; self.pulses = None
        self.setWindowTitle('Paste exact per-bin RF powers and times')
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            'Columns: bin_index, rate, start, duration (optional fifth column: enabled, 0 or 1). Bin indices are 0-based. '
            'Use an R,rate,start,duration header to enter physical R instead. '
            'Repeated bins are allowed. This replaces only the selected profile; other profiles are unchanged.')
        label.setWordWrap(True); layout.addWidget(label)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setPlaceholderText('bin_index,rate,start,duration\n240,1.35,0.0,0.25\n241,2.10,0.0,0.40\n242,0.80,0.1,0.30')
        layout.addWidget(self.text, 1)
        self.error_label = QtWidgets.QLabel(''); self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = QtWidgets.QHBoxLayout(); buttons.addStretch()
        cancel = QtWidgets.QPushButton('Cancel'); cancel.clicked.connect(self.reject)
        apply = QtWidgets.QPushButton('Replace profile with pasted values'); apply.clicked.connect(self._commit)
        buttons.addWidget(cancel); buttons.addWidget(apply); layout.addLayout(buttons)
        self.resize(650, 390)

    def _commit(self):
        try: self.pulses = parse_custom_table(self.text.toPlainText(), self.grid)
        except ValueError as exc:
            self.error_label.setText(str(exc)); return
        self.accept()


class ProfileEditor(QtWidgets.QDialog):
    def __init__(self, program: PulseProgram, monitor_R: float = 0.4, parent=None):
        super().__init__(parent)
        self.program = program.clone()
        self.monitor_R = monitor_R
        self.current_profile = -1
        self.setWindowTitle('Ideal RF profiles — numerical power and time for EVERY bin')
        self.resize(980, 565)
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(7,7,7,7); layout.setSpacing(5)
        hint = QtWidgets.QLabel(
            'Each row is one bin pulse. Use its arrows, or select rows and click Enter exact values. '
            'Scroll wheels never change numerical fields. Times are relative to Start program. Overlapping pulses ADD.')
        hint.setWordWrap(True); layout.addWidget(hint)
        body = QtWidgets.QHBoxLayout(); layout.addLayout(body,1)
        left = QtWidgets.QVBoxLayout(); body.addLayout(left)
        self.profile_list = QtWidgets.QListWidget(); self.profile_list.setMaximumWidth(180)
        self.profile_list.setMinimumWidth(135)
        self.profile_list.currentRowChanged.connect(self._select_profile)
        self.profile_list.itemChanged.connect(self._enable_profile)
        left.addWidget(self.profile_list,1)
        for text, callback in [('Add custom profile',self._add_profile),('Duplicate profile',self._duplicate),
                               ('Remove profile',self._remove),('Load horn / pedestal example',self._confirm_example)]:
            button=QtWidgets.QPushButton(text); button.clicked.connect(callback); left.addWidget(button)
        right=QtWidgets.QVBoxLayout(); body.addLayout(right,1)
        name_row=QtWidgets.QHBoxLayout(); name_row.addWidget(QtWidgets.QLabel('Profile name'))
        self.name_box=QtWidgets.QLineEdit(); self.name_box.editingFinished.connect(self._rename)
        name_row.addWidget(self.name_box,1); right.addLayout(name_row)

        region_box=QtWidgets.QGroupBox('Profile range: L = left bin, R = right bin (inclusive)')
        region=QtWidgets.QGridLayout(region_box); region.setContentsMargins(6,6,6,6)
        step=float(self.program.grid[1]-self.program.grid[0])
        self.left_R=number_box(-0.98,program.r_min,program.r_max,step=step)
        self.right_R=number_box(-0.72,program.r_min,program.r_max,step=step)
        self.left_R.setObjectName('left_bound_R'); self.right_R.setObjectName('right_bound_R')
        region.addWidget(QtWidgets.QLabel('Left L (R)'),0,0); region.addWidget(self.left_R,0,1)
        region.addWidget(QtWidgets.QLabel('Right R (R)'),0,2); region.addWidget(self.right_R,0,3)
        set_bounds=QtWidgets.QPushButton('Enter bounds...'); set_bounds.clicked.connect(self._exact_bounds)
        region.addWidget(set_bounds,0,4)
        self.range_label=QtWidgets.QLabel(); self.range_label.setWordWrap(True)
        region.addWidget(self.range_label,1,0,1,5)
        self.left_R.valueChanged.connect(self._update_range_label)
        self.right_R.valueChanged.connect(self._update_range_label)
        right.addWidget(region_box)

        self.tabs=QtWidgets.QTabWidget(); right.addWidget(self.tabs,1)
        custom=QtWidgets.QWidget(); custom_layout=QtWidgets.QVBoxLayout(custom)
        custom_layout.setContentsMargins(2,4,2,2)
        row=QtWidgets.QHBoxLayout()
        self.ensure_button=QtWidgets.QPushButton('Add every missing bin in L..R (U=0)')
        self.ensure_button.clicked.connect(self._ensure_bins)
        self.exact_button=QtWidgets.QPushButton('Enter exact values for selected rows...')
        self.exact_button.clicked.connect(self._exact_rows)
        row.addWidget(self.ensure_button); row.addWidget(self.exact_button); custom_layout.addLayout(row)
        self.table=QtWidgets.QTableWidget(0,7)
        self.table.setHorizontalHeaderLabels(['On','Bin (0-based)','Physical R','RF power U','Start time','Duration','Stop time'])
        self.table.setSortingEnabled(False); self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(SELECTION.SelectRows)
        mode=VIEW.SelectionMode if hasattr(VIEW,'SelectionMode') else VIEW
        self.table.setSelectionMode(mode.ExtendedSelection)
        # Permanent arrow editors are explicit; ordinary clicks cannot start typing.
        self.table.setEditTriggers(EDIT.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._double_click)
        header=self.table.horizontalHeader()
        resize=QtWidgets.QHeaderView.ResizeMode if hasattr(QtWidgets.QHeaderView,'ResizeMode') else QtWidgets.QHeaderView
        header.setSectionResizeMode(resize.Stretch); header.setSectionResizeMode(0,resize.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.itemChanged.connect(self._cell_changed)
        custom_layout.addWidget(self.table,1)
        row_buttons=QtWidgets.QHBoxLayout()
        for text,fun in [('Add pulse at monitor',self._add_pulse),('Duplicate rows',self._duplicate_rows),
                         ('Delete selected',self._delete_rows),('Select L..R rows',self._select_region_rows)]:
            button=QtWidgets.QPushButton(text);button.clicked.connect(fun);row_buttons.addWidget(button)
        custom_layout.addLayout(row_buttons)
        paste_row=QtWidgets.QHBoxLayout()
        for text,fun in [('Paste custom table...',self._paste_table),('Copy table',self._copy_table)]:
            button=QtWidgets.QPushButton(text);button.clicked.connect(fun);paste_row.addWidget(button)
        paste_hint=QtWidgets.QLabel('RF power U is the existing base-rate setting, not watts.')
        paste_hint.setWordWrap(True);paste_row.addWidget(paste_hint,1);custom_layout.addLayout(paste_row)
        self.tabs.addTab(custom,'Per-bin numerical values')

        presets=QtWidgets.QWidget(); gen=QtWidgets.QGridLayout(presets)
        self.shape_box=NoWheelComboBox()
        for label,key in [('Flat','flat'),('Linear ramp','linear'),('Triangle','triangle'),('Gaussian (cut at bounds)','gaussian')]:
            self.shape_box.addItem(label,key)
        self.rate_box=number_box(2.0);self.end_rate_box=number_box(2.0)
        self.start_box=number_box(0.0);self.duration_box=number_box(1.0)
        self.start_step_box=number_box(0.0,step=0.01);self.end_duration_box=number_box(1.0)
        fields=[('Shape',self.shape_box),('Power / peak U',self.rate_box),('Right power (ramp)',self.end_rate_box),
                ('Start time',self.start_box),('Duration left',self.duration_box),('Duration right',self.end_duration_box),
                ('Start step per bin',self.start_step_box)]
        for i,(label,widget) in enumerate(fields):
            r,c=divmod(i,3);gen.addWidget(QtWidgets.QLabel(label),2*r,c);gen.addWidget(widget,2*r+1,c)
        explain=QtWidgets.QLabel('Presets are optional. They replace the selected profile only; the generated powers and times can then be changed in every individual row.')
        explain.setWordWrap(True);gen.addWidget(explain,6,0,1,3)
        generate=QtWidgets.QPushButton('Replace profile with preset in L..R')
        generate.clicked.connect(self._confirm_generate);gen.addWidget(generate,7,0,1,3)
        gen.setRowStretch(8,1)
        self.tabs.addTab(presets,'Optional shape presets')
        self.summary=QtWidgets.QLabel(); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        io=QtWidgets.QHBoxLayout()
        for text,fun in [('Load JSON / CSV',self._load),('Save JSON',self._save_json),('Export CSV',self._save_csv)]:
            b=QtWidgets.QPushButton(text);b.clicked.connect(fun);io.addWidget(b)
        io.addStretch()
        cancel=QtWidgets.QPushButton('Cancel');cancel.clicked.connect(self.reject)
        apply_button=QtWidgets.QPushButton('Apply program');apply_button.clicked.connect(self._apply)
        apply_button.setDefault(False)
        io.addWidget(cancel);io.addWidget(apply_button);layout.addLayout(io)
        self._refresh_list(0)
        screen=QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            size=screen.availableGeometry()
            self.resize(min(980,int(.95*size.width())),min(565,int(.88*size.height())))

    def _error(self,exc): QtWidgets.QMessageBox.warning(self,'RF program',str(exc))

    def _profile(self):
        if 0 <= self.current_profile < len(self.program.profiles): return self.program.profiles[self.current_profile]
        return None

    def _refresh_list(self,selected=0):
        with blocked(self.profile_list):
            self.profile_list.clear()
            for p in self.program.profiles:
                item=QtWidgets.QListWidgetItem(p.name)
                item.setFlags(item.flags() | FLAGS.ItemIsUserCheckable)
                item.setCheckState(CHECK.Checked if p.enabled else CHECK.Unchecked)
                self.profile_list.addItem(item)
            selected=min(max(0,selected),len(self.program.profiles)-1)
            self.profile_list.setCurrentRow(selected)
        self._select_profile(selected)

    def _select_profile(self,index):
        self.current_profile=index;p=self._profile()
        with blocked(self.name_box): self.name_box.setText('' if p is None else p.name)
        if p is not None and p.pulses:
            bins=[x.bin_index for x in p.pulses]
            with blocked(self.left_R): self.left_R.setValue(float(self.program.grid[min(bins)]))
            with blocked(self.right_R): self.right_R.setValue(float(self.program.grid[max(bins)]))
        self._update_range_label();self._render_table()

    def _update_range_label(self,*args):
        try:
            bins=region_indices(self.program.grid,self.left_R.value(),self.right_R.value())
            self.range_label.setText(f'Bins {bins.start}..{bins.stop-1} ({len(bins)} bins). '
                f'Snapped centers: {self.program.grid[bins.start]:+.6f} .. {self.program.grid[bins.stop-1]:+.6f}. '
                'Bounds select rows to create; they do not move or stretch existing pulses.')
        except ValueError as exc: self.range_label.setText(str(exc))

    def _enable_profile(self,item):
        i=self.profile_list.row(item)
        if i>=0:
            self.program.profiles[i].enabled=item.checkState()==CHECK.Checked;self._summary()

    def _rename(self):
        p=self._profile()
        if p is not None:
            name=self.name_box.text().strip()
            if not name:self.name_box.setText(p.name);return
            p.name=name
            with blocked(self.profile_list):self.profile_list.item(self.current_profile).setText(name)

    def _render_table(self):
        p=self._profile();pulses=[] if p is None else p.pulses;grid=self.program.grid
        self.table.setUpdatesEnabled(False)
        try:
            with blocked(self.table):
                self.table.setRowCount(0);self.table.setRowCount(len(pulses))
                for row,pulse in enumerate(pulses):
                    item=QtWidgets.QTableWidgetItem()
                    item.setFlags(FLAGS.ItemIsSelectable|FLAGS.ItemIsEnabled|FLAGS.ItemIsUserCheckable)
                    item.setCheckState(CHECK.Checked if pulse.enabled else CHECK.Unchecked)
                    self.table.setItem(row,0,item)
                    values=[str(pulse.bin_index),f'{grid[pulse.bin_index]:.9g}',repr(pulse.rate),repr(pulse.start),repr(pulse.duration),f'{pulse.stop:.12g}']
                    for col,text in enumerate(values,1):
                        item=QtWidgets.QTableWidgetItem(text)
                        item.setFlags(FLAGS.ItemIsSelectable|FLAGS.ItemIsEnabled)
                        self.table.setItem(row,col,item)
                    # Each bin has explicit numerical controls; presets are not required.
                    bin_box=ArrowSpinBox();bin_box.setRange(0,self.program.n_bins-1);bin_box.setValue(pulse.bin_index)
                    bin_box.setObjectName(f'pulse_{row}_bin')
                    bin_box.valueChanged.connect(lambda v,r=row:self._arrow_cell(r,1,v))
                    self.table.setCellWidget(row,1,bin_box)
                    for col,value,step in [(3,pulse.rate,.1),(4,pulse.start,.01),(5,pulse.duration,.01)]:
                        box=number_box(value,hi=max(1e6,value),decimals=9,step=step)
                        box.setObjectName(f'pulse_{row}_{col}')
                        box.valueChanged.connect(lambda v,r=row,c=col:self._arrow_cell(r,c,v))
                        self.table.setCellWidget(row,col,box)
        finally: self.table.setUpdatesEnabled(True)
        self._summary()

    def _arrow_cell(self,row,col,value):
        # Preserve selection unless the user steps a row outside it.
        if row not in self._selected_rows():self.table.selectRow(row)
        self.table.item(row,col).setText(str(int(value)) if col==1 else repr(float(value)))

    def _cell_changed(self,item):
        profile=self._profile()
        if profile is None:return
        row=item.row()
        if row>=len(profile.pulses):return
        try:
            pulse=BinPulse(int(self.table.item(row,1).text()),float(self.table.item(row,3).text()),
                           float(self.table.item(row,4).text()),float(self.table.item(row,5).text()),
                           self.table.item(row,0).checkState()==CHECK.Checked)
            pulse.validate(self.program.n_bins);profile.pulses[row]=pulse
            with blocked(self.table):
                self.table.item(row,2).setText(f'{self.program.grid[pulse.bin_index]:.9g}')
                self.table.item(row,6).setText(f'{pulse.stop:.12g}')
            for col,value in [(1,pulse.bin_index),(3,pulse.rate),(4,pulse.start),(5,pulse.duration)]:
                box=self.table.cellWidget(row,col)
                if box is not None:
                    with blocked(box):
                        if value>box.maximum():box.setMaximum(value)
                        box.setValue(value)
            self._summary()
        except (ValueError,TypeError,OverflowError) as exc:self._error(exc);self._render_table()

    def _summary(self):
        try:
            c=self.program.compile()
            self.summary.setText(f'{sum(p.enabled for p in self.program.profiles)} enabled profiles; '
                f'{len(c.bins)} active pulse rows; {np.count_nonzero(c.envelope)} addressed bins; '
                f'ends at t={c.end_time:.6g}. Largest combined base rate: {self.program.gain*float(c.envelope.max()):.6g}. '
                'Zero power or duration gives no irradiation. Apply program commits; Cancel leaves the running program unchanged.')
        except ValueError as exc:self.summary.setText(str(exc))

    def _selected_rows(self):
        return sorted({x.row() for x in self.table.selectionModel().selectedRows()})

    def _restore_selection(self,rows):
        # selectRow alone clears old selections; use the selection model flags.
        flag=QtCore.QItemSelectionModel.SelectionFlag if hasattr(QtCore.QItemSelectionModel,'SelectionFlag') else QtCore.QItemSelectionModel
        for row in rows:
            if row<self.table.rowCount():self.table.selectionModel().select(self.table.model().index(row,0),flag.Select|flag.Rows)

    def apply_exact_values(self,rows,**values):
        p=self._profile()
        if p is None:raise ValueError('Create or select a profile first.')
        update_rows(p,rows,self.program.n_bins,**values)
        self._render_table();self._restore_selection(rows)

    def _exact_rows(self,*args):
        rows=self._selected_rows();p=self._profile()
        if p is None or not rows:self._error('Select one or more rows in the bin table first.');return
        pulse=p.pulses[rows[0]]
        fields=[('rate','RF power U',pulse.rate,0,float('inf')),
                ('start','Start time',pulse.start,0,float('inf')),
                ('duration','Duration',pulse.duration,0,float('inf'))]
        d=ExactValuesDialog('Exact values for selected bin pulses',fields,self,optional=True,
            explanation=f'{len(rows)} row(s) selected. Checked values are applied to each selected row. '
                        'Uncheck a value to preserve it. Times use simulation units.')
        if len(rows) > 1:
            for check in d.checks.values(): check.setChecked(False)
        if accepted(d,run_dialog(d)):
            try:self.apply_exact_values(rows,**d.values)
            except ValueError as exc:self._error(exc)

    def _double_click(self,row,col):
        if col in (2,6):return
        self.table.selectRow(row);self._exact_rows()

    def _exact_bounds(self):
        d=ExactValuesDialog('Set profile creation bounds',[
            ('left','Left L (physical R)',self.left_R.value(),self.program.r_min,self.program.r_max),
            ('right','Right R (physical R)',self.right_R.value(),self.program.r_min,self.program.r_max)],self,
            explanation='Values snap to existing frequency-bin centers. This only selects a range; it does not yet change any pulse.')
        if accepted(d,run_dialog(d)):
            try:
                bins=region_indices(self.program.grid,d.values['left'],d.values['right'])
                self.left_R.setValue(float(self.program.grid[bins.start]))
                self.right_R.setValue(float(self.program.grid[bins.stop-1]))
            except ValueError as exc:self._error(exc)

    def _ensure_bins(self):
        if self._profile() is None:self._add_profile()
        try:
            ensure_region_rows(self._profile(),self.program.grid,self.left_R.value(),self.right_R.value(),
                               start=self.start_box.value(),duration=self.duration_box.value())
            self._render_table();self.tabs.setCurrentIndex(0);self._select_region_rows()
        except ValueError as exc:self._error(exc)

    def _select_region_rows(self):
        p=self._profile()
        if p is None:return
        try:bins=region_indices(self.program.grid,self.left_R.value(),self.right_R.value())
        except ValueError as exc:self._error(exc);return
        rows=[i for i,x in enumerate(p.pulses) if x.bin_index in bins]
        self.table.clearSelection();self._restore_selection(rows)
        if rows:self.table.scrollToItem(self.table.item(rows[0],2))

    def _add_profile(self):
        self.program.profiles.append(RFProfile(f'Custom profile {len(self.program.profiles)+1}'))
        self._refresh_list(len(self.program.profiles)-1);self.tabs.setCurrentIndex(0)

    def _duplicate(self):
        p=self._profile()
        if p is None:return
        self.program.profiles.append(RFProfile(p.name+' copy',p.enabled,[BinPulse(**vars(x)) for x in p.pulses]))
        self._refresh_list(len(self.program.profiles)-1)

    def _remove(self):
        if self._profile() is not None:
            del self.program.profiles[self.current_profile];self._refresh_list(self.current_profile)

    def _confirm(self,message):
        buttons=QtWidgets.QMessageBox.StandardButton if hasattr(QtWidgets.QMessageBox,'StandardButton') else QtWidgets.QMessageBox
        return QtWidgets.QMessageBox.question(self,'Replace RF rows?',message,buttons.Yes|buttons.No,buttons.No)==buttons.Yes

    def _confirm_example(self):
        if self._confirm('Replace the editor program with the example? Your running program is not changed until Apply program.'):
            self._example()

    def _example(self):
        self.program=example_program(self.program.grid);self._refresh_list(0);self.tabs.setCurrentIndex(0)

    def _confirm_generate(self):
        p=self._profile()
        if p is None or not p.pulses or self._confirm('Replace all rows of this profile with the preset? Individual edits in this profile will be overwritten.'):
            self._generate()

    def _generate(self):
        if self._profile() is None:self._add_profile()
        try:
            old=self._profile()
            p=make_profile(self.program.grid,old.name,self.left_R.value(),self.right_R.value(),
                           self.shape_box.currentData(),self.rate_box.value(),self.end_rate_box.value(),
                           self.start_box.value(),self.duration_box.value(),self.start_step_box.value(),self.end_duration_box.value())
            p.enabled=old.enabled;self.program.profiles[self.current_profile]=p
            self._render_table();self.tabs.setCurrentIndex(0)
        except ValueError as exc:self._error(exc)

    def _add_pulse(self):
        if self._profile() is None:self._add_profile()
        b=nearest_bin(self.program.grid,self.monitor_R)
        self._profile().pulses.append(BinPulse(b,2,0,1));self._render_table()
        self.table.selectRow(self.table.rowCount()-1);self.table.scrollToBottom()

    def _delete_rows(self):
        p=self._profile()
        if p is None:return
        for row in reversed(self._selected_rows()):del p.pulses[row]
        self._render_table()

    def _duplicate_rows(self):
        p=self._profile()
        if p is None:return
        p.pulses.extend([BinPulse(**vars(p.pulses[i])) for i in self._selected_rows()]);self._render_table()

    def _paste_table(self):
        d=PasteTableDialog(self.program.grid,self)
        if accepted(d,run_dialog(d)):
            if self._profile() is None:self._add_profile()
            self._profile().pulses=d.pulses;self._select_profile(self.current_profile);self.tabs.setCurrentIndex(0)

    def _copy_table(self):
        p=self._profile()
        if p is not None:QtWidgets.QApplication.clipboard().setText(custom_table_text(p.pulses))

    def _load(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Load RF program','','RF programs (*.json *.csv)')
        if not path:return
        try:
            p=PulseProgram.load_csv(path) if Path(path).suffix.lower()=='.csv' else PulseProgram.load(path)
            p.validate_grid(self.program.grid);self.program=p;self._refresh_list(0)
        except (ValueError,OSError) as exc:self._error(exc)

    def _save_json(self):
        self._rename()
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save RF program','rf_program.json','JSON (*.json)')
        if path:
            try:self.program.save(path)
            except (ValueError,OSError) as exc:self._error(exc)

    def _save_csv(self):
        self._rename()
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Export all bin rows','rf_program.csv','CSV (*.csv)')
        if path:
            try:self.program.save_csv(path)
            except (ValueError,OSError) as exc:self._error(exc)

    def _apply(self):
        try:self._rename();self.program.validate();self.program.compile();self.accept()
        except ValueError as exc:self._error(exc)
