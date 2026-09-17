"""Explicit, confirmed typed entry. Live spin fields themselves are arrow-only."""
from __future__ import annotations
import math
from .baseline_gui import QtWidgets


def accepted(dialog, result):
    code = QtWidgets.QDialog.DialogCode.Accepted if hasattr(QtWidgets.QDialog, 'DialogCode') else QtWidgets.QDialog.Accepted
    return result == code


def run_dialog(dialog):
    return dialog.exec() if hasattr(dialog, 'exec') else dialog.exec_()


class ExactValuesDialog(QtWidgets.QDialog):
    """Enter named values and explicitly commit, optionally choosing a subset."""
    def __init__(self, title, fields, parent=None, *, optional=False, explanation=''):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.values = {}
        self.edits = {}
        self.checks = {}
        self.fields = fields  # (key, label, value, minimum, maximum)
        layout = QtWidgets.QVBoxLayout(self)
        if explanation:
            label = QtWidgets.QLabel(explanation); label.setWordWrap(True)
            layout.addWidget(label)
        form = QtWidgets.QGridLayout()
        for i, (key, label, value, low, high) in enumerate(fields):
            if optional:
                check = QtWidgets.QCheckBox(label); check.setChecked(True)
                self.checks[key] = check
                form.addWidget(check, i, 0)
            else:
                form.addWidget(QtWidgets.QLabel(label), i, 0)
            edit = QtWidgets.QLineEdit(repr(float(value)))
            edit.setObjectName(f'exact_{key}')
            edit.setToolTip('Type a finite numerical value; scientific notation is accepted. Nothing changes until Apply.')
            self.edits[key] = edit
            form.addWidget(edit, i, 1)
            if optional: self.checks[key].toggled.connect(edit.setEnabled)
        layout.addLayout(form)
        self.error_label = QtWidgets.QLabel(''); self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        row = QtWidgets.QHBoxLayout(); row.addStretch()
        cancel = QtWidgets.QPushButton('Cancel'); cancel.clicked.connect(self.reject)
        apply = QtWidgets.QPushButton('Apply exact values'); apply.clicked.connect(self._commit)
        apply.setDefault(True)
        row.addWidget(cancel); row.addWidget(apply); layout.addLayout(row)
        self.resize(480, self.sizeHint().height())

    def _commit(self):
        try:
            values = {}
            for key, label, value, low, high in self.fields:
                if key in self.checks and not self.checks[key].isChecked(): continue
                number = float(self.edits[key].text().strip())
                if not math.isfinite(number) or number < low or number > high:
                    raise ValueError(f'{label}: enter a finite value between {low:g} and {high:g}.')
                values[key] = number
            if not values: raise ValueError('Check at least one value to apply.')
        except (ValueError, OverflowError) as exc:
            self.error_label.setText(str(exc)); return
        self.values = values
        self.accept()
