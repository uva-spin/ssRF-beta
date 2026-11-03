"""Typed numeric fields with bounds checking and no stepping buttons.

Commit on Enter or focus loss, never on intermediate keystrokes. Retain Qt's
numeric validation and programmatic setValue API; ignore wheel/arrow stepping.
"""
from __future__ import annotations
from .baseline_gui import QtCore, QtWidgets

QT = QtCore.Qt
KEYS = QT.Key if hasattr(QT, 'Key') else QT
EVENT = QtCore.QEvent.Type if hasattr(QtCore.QEvent, 'Type') else QtCore.QEvent


class _TypedNumberMixin:
    def _configure_entry(self):
        self.setKeyboardTracking(False)
        self.setAccelerated(False)
        symbols = QtWidgets.QAbstractSpinBox.ButtonSymbols if hasattr(QtWidgets.QAbstractSpinBox, 'ButtonSymbols') else QtWidgets.QAbstractSpinBox
        self.setButtonSymbols(symbols.NoButtons)
        self.lineEdit().setReadOnly(False)
        self.lineEdit().setAcceptDrops(False)
        self.setAcceptDrops(False)
        self.lineEdit().installEventFilter(self)
        self.setToolTip('Type a value, then press Enter or leave the field to apply it. Scrolling does not change the value.')

    def stepEnabled(self):
        flags = QtWidgets.QAbstractSpinBox.StepEnabledFlag if hasattr(QtWidgets.QAbstractSpinBox, 'StepEnabledFlag') else QtWidgets.QAbstractSpinBox
        return flags.StepNone

    def wheelEvent(self, event):
        # Ignore, rather than consume, so an enclosing control panel can scroll.
        event.ignore()

    def keyPressEvent(self, event):
        # Enter commits this field without activating a dialog's default button.
        if event.key() in (KEYS.Key_Return, KEYS.Key_Enter):
            self.interpretText()
            self.editingFinished.emit()
            event.accept()
            return
        if event.key() in (KEYS.Key_Up, KEYS.Key_Down, KEYS.Key_PageUp, KEYS.Key_PageDown):
            event.ignore()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if obj is self.lineEdit():
            if event.type() == EVENT.Wheel:
                event.ignore()
                return True
            if event.type() == EVENT.KeyPress and event.key() in (
                    KEYS.Key_Up, KEYS.Key_Down, KEYS.Key_PageUp, KEYS.Key_PageDown):
                event.ignore()
                return True
        return super().eventFilter(obj, event)


class NumericDoubleEdit(_TypedNumberMixin, QtWidgets.QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._configure_entry()


class NumericIntEdit(_TypedNumberMixin, QtWidgets.QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._configure_entry()


# Compatibility for scripts importing the earlier widget names.
ArrowDoubleSpinBox = NumericDoubleEdit
ArrowSpinBox = NumericIntEdit


class NoWheelComboBox(QtWidgets.QComboBox):
    def wheelEvent(self, event):
        # Choose an item explicitly from the dropdown instead.
        event.ignore()
