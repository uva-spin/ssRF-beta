"""Arrow-only GUI number controls, with no mouse-wheel value changes.

The QAbstractSpinBox itself is NOT set read-only (that would disable its
buttons); only its embedded QLineEdit is. Exact typed entry is offered by
explicit dialogs elsewhere. Programmatic setValue remains available.
"""
from __future__ import annotations
from .baseline_gui import QtCore, QtWidgets

QT = QtCore.Qt
KEYS = QT.Key if hasattr(QT, 'Key') else QT
EVENT = QtCore.QEvent.Type if hasattr(QtCore.QEvent, 'Type') else QtCore.QEvent


class _ArrowOnlyMixin:
    def _protect(self):
        self.setKeyboardTracking(False)
        self.setAccelerated(False)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setAcceptDrops(False)
        self.setAcceptDrops(False)
        self.lineEdit().installEventFilter(self)
        self.setToolTip('Use the on-screen up/down arrows. Scrolling and typing do not change this field.')

    def wheelEvent(self, event):
        # Ignore, rather than consume, so an enclosing control panel can scroll.
        event.ignore()

    def keyPressEvent(self, event):
        # Keyboard stepping is disabled too: adjustment is by visible GUI arrows.
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


class ArrowDoubleSpinBox(_ArrowOnlyMixin, QtWidgets.QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._protect()


class ArrowSpinBox(_ArrowOnlyMixin, QtWidgets.QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._protect()


class NoWheelComboBox(QtWidgets.QComboBox):
    def wheelEvent(self, event):
        # Choose an item explicitly from the dropdown instead.
        event.ignore()
