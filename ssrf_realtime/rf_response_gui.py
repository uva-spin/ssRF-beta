"""Preview and apply independent material RF settings without editing live state."""
import copy
import numpy as np
from .baseline_gui import QtWidgets, Figure, FigureCanvas
from .safe_widgets import NoWheelComboBox
from .material_config import MaterialConfig


FIELDS = (
    ('rf_pedestal_gain', 'Pedestal RF gain'),
    ('rf_density_power', 'RF density exponent'),
    ('rf_density_clip', 'RF density weight clip'),
)
PEAK_FIELDS = (
    ('rf_peak_center_R', 'Peak-band centers: ±R'),
    ('rf_peak_half_width_R', 'Peak-band half-width (R)'),
    ('rf_transition_width_R', 'Smooth transition width (R)'),
)

SATURATION_FIELDS = (
    ('rf_spectral_strength', 'RF spectral strength [R / (time × U)]'),
    ('rf_homogeneous_hwhm_R', 'Homogeneous RF HWHM [R]'),
)
RECOVERY_FIELDS = (
    ('diffusion_scale', 'Exchange scale K0 [1 / time]'),
    ('zq_width_R', 'Exchange overlap width [R]'),
    ('cross_branch_ratio', 'Cross-branch / same-branch ratio'),
    ('double_quantum_ratio', 'DQ / same-branch ratio'),
    ('orientation_corr_fraction', 'Orientation correlation fraction [0–1]'),
    ('orientation_corr_width_deg', 'Orientation correlation width [degrees]'),
)


class RFResponseDialog(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle('ssRF — RF saturation and spin diffusion')
        self.original = config.to_dict()
        self.value = None
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            'Peak selectivity restores adjustable contrast between overlapping RF transitions. '
            'It is an empirical material response, separate from spin diffusion. '
            'Apply keeps populations, AFP and budget settings and pauses RF.')
        hint.setWordWrap(True); layout.addWidget(hint)
        form = QtWidgets.QFormLayout()
        self.mode = NoWheelComboBox()
        self.mode.addItem('Budget compatibility — shared RF/DNP density weighting', 'legacy')
        self.mode.addItem('Independent RF — tune pedestal sensitivity', 'independent')
        self.mode.addItem('Spectral saturation — adjustable peak selectivity', 'spectral')
        data = self.original['parameters']['rf_response']
        # An old material starts independent mode from its own density law.
        if data['rf_response_mode'] == 'legacy':
            coupling = self.original['parameters']['rf_coupling']
            data['rf_density_power'] = coupling['capacity_rate_power']
            data['rf_density_clip'] = coupling['capacity_rate_clip']
        self.mode.setCurrentIndex(self.mode.findData(data['rf_response_mode']))
        form.addRow('Response model', self.mode)
        layout.addLayout(form)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        rf_page = QtWidgets.QWidget(); page_layout = QtWidgets.QVBoxLayout(rf_page)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        contents = QtWidgets.QWidget(); rf_layout = QtWidgets.QVBoxLayout(contents)
        scroll.setWidget(contents); page_layout.addWidget(scroll)
        self.tabs.addTab(rf_page, 'RF saturation')
        self.edits = {}
        self.edit_groups = {}
        self.rf_forms = {}
        for group, fields in [('rf_saturation', SATURATION_FIELDS),
                              ('rf_selectivity', (('rf_peak_selectivity', 'Peak selectivity [0–1]'),)),
                              ('rf_response', FIELDS)]:
            container = QtWidgets.QWidget(); group_form = QtWidgets.QFormLayout(container)
            self.rf_forms[group] = container; rf_layout.addWidget(container)
            self._add_fields(group_form, group, fields)
        container = QtWidgets.QWidget(); group_form = QtWidgets.QFormLayout(container)
        self.rf_forms['peak_region'] = container; rf_layout.addWidget(container)
        self._add_fields(group_form, 'rf_response', PEAK_FIELDS)
        self.rf_hint = QtWidgets.QLabel(); self.rf_hint.setWordWrap(True)
        rf_layout.addWidget(self.rf_hint); rf_layout.addStretch()
        recovery_page = QtWidgets.QWidget(); recovery_form = QtWidgets.QFormLayout(recovery_page)
        self.tabs.addTab(recovery_page, 'Spin diffusion')
        self._add_fields(recovery_form, 'recovery', RECOVERY_FIELDS)
        self.overlap = NoWheelComboBox()
        for value in ('lorentzian','gaussian'): self.overlap.addItem(value,value)
        self.overlap.setCurrentIndex(self.overlap.findData(self.original['parameters']['recovery']['diffusion_overlap']))
        recovery_form.addRow('Exchange overlap shape', self.overlap)
        recovery_hint = QtWidgets.QLabel(
            'Pair exchange redistributes the current populations and conserves total vector polarization. '
            'K0 and the bandwidth affect RF-on replenishment AND RF-off recovery. '
            'Orientation correlation is an effective geometry assumption, not a crystal structure. '
            'The master Spin diffusion switch is in the main window. T1 and DNP remain separate.')
        recovery_hint.setWordWrap(True); recovery_form.addRow(recovery_hint)
        self.figure = Figure(figsize=(6, 2.2), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(160)
        layout.addWidget(self.canvas)
        self.error = QtWidgets.QLabel(''); self.error.setWordWrap(True); layout.addWidget(self.error)
        row = QtWidgets.QHBoxLayout()
        for title, fun in [('Preview response', self.preview), ('Cancel', self.reject), ('Apply material response', self.commit)]:
            button = QtWidgets.QPushButton(title); button.clicked.connect(fun); row.addWidget(button)
        layout.addLayout(row)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.tabs.currentChanged.connect(self.preview)
        self._mode_changed()
        self.resize(790, 820)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            self.resize(min(790, int(screen.availableGeometry().width()*.9)),
                        min(820, int(screen.availableGeometry().height()*.9)))

    def _add_fields(self, form, group, fields):
        for key, title in fields:
            edit = QtWidgets.QLineEdit(repr(self.original['parameters'][group][key]))
            edit.setObjectName(key); self.edits[key] = edit; self.edit_groups[key] = group
            form.addRow(title, edit)

    def candidate(self):
        data = copy.deepcopy(self.original)
        response = data['parameters']['rf_response']
        response['rf_response_mode'] = self.mode.currentData()
        for key, edit in self.edits.items():
            data['parameters'][self.edit_groups[key]][key] = float(edit.text().strip())
        data['parameters']['recovery']['diffusion_overlap'] = self.overlap.currentData()
        return MaterialConfig.from_dict(data)

    def _mode_changed(self, *args):
        spectral = self.mode.currentData() == 'spectral'
        self.rf_forms['rf_saturation'].setVisible(spectral)
        self.rf_forms['rf_selectivity'].setVisible(spectral)
        self.rf_forms['rf_response'].setVisible(not spectral)
        self.rf_forms['rf_response'].setEnabled(self.mode.currentData() == 'independent')
        self.rf_forms['peak_region'].setEnabled(self.mode.currentData() != 'legacy')
        self.rf_hint.setText(
            'Selectivity 0 gives equal per-spin rates; 0.7 is the default empirical setting; '
            '1 gives Budget-like relative coupling inside the peak bands. Higher values reduce the '
            'weaker packet response there. The densest packet and outer pedestal coupling stay fixed. '
            'This is empirical, not a microscopic density law. U remains uncalibrated carrier power.'
            if spectral else 'Compatibility mode: the density/pedestal multiplier is an empirical RF response. '
            'These settings are ignored in spectral saturation mode.')
        self.preview()

    def preview(self, *args):
        try:
            model = self.candidate().make_model(preserve_state=False)
        except (ValueError, TypeError, OverflowError) as exc:
            self.error.setText(str(exc)); return
        self.figure.clear(); ax = self.figure.add_subplot(111)
        p = model.params
        if self.tabs.currentIndex() == 1:
            connectivity = model.diffusion_connectivity()
            for key, title in [('same_plus0','Same-branch SQ'),('cross_plus','Cross-branch SQ'),
                               ('double_quantum','DQ')]:
                ax.plot(model.Rplus,connectivity[key],label=title)
            ax.set(xlabel='Packet coordinate R (plus branch)',ylabel='Connectivity [1 / time]',
                   title='Exchange connectivity — actual refill also depends on populations')
            ax.legend(fontsize=8)
            detail = 'These are maximum connectivity weights, not measured recovery rates.'
            if not p.diffusion_enabled:detail += ' Spin diffusion is currently switched off.'
        elif p.rf_response_mode == 'spectral':
            response = model.rf_packet_response()
            ax.plot(model.Rplus, response, label='I+ packets')
            ax.plot(model.Rplus, response[::-1], label='I− packets')
            ax.axhline(1., color='gray', ls=':', lw=1)
            ax.set(xlabel='Physical frequency coordinate R', ylabel='Relative RF coupling',
                   title='Peak selectivity — relative to equal per-spin rates')
            ax.legend(loc='lower right', fontsize=8)
            detail = (f'Unmodified on-resonance pair rate per U: {p.rf_spectral_strength/p.rf_homogeneous_hwhm_R:.5g}. '
                      f'RF HWHM / grid spacing: {p.rf_homogeneous_hwhm_R/model.dR:.2f}; refine the grid to check narrow features.')
        else:
            ax.plot(model.Rplus, model.rf_pedestal_profile(), color='#146a9e')
            ax.set(xlabel='Physical frequency coordinate R', ylabel='Pedestal gain',
                   title='Empirical RF gain relative to the selected density response')
            for center in (-p.rf_peak_center_R, p.rf_peak_center_R):
                ax.axvspan(center-p.rf_peak_half_width_R,center+p.rf_peak_half_width_R,color='gray',alpha=.16)
            detail = 'Shaded bands protect the empirical peak RF response.'
        ax.grid(alpha=.2); self.canvas.draw()
        self.error.setText(detail+' Changes apply only when you click Apply.')

    def commit(self, *args):
        try:
            self.value = self.candidate()
        except (ValueError, TypeError, OverflowError) as exc:
            self.error.setText(str(exc)); return
        self.accept()
