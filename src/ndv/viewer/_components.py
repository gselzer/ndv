from __future__ import annotations

from enum import Enum
from pathlib import Path

from qtpy.QtCore import QSize
from qtpy.QtGui import QMovie
from qtpy.QtWidgets import QLabel, QPushButton, QWidget
from superqt import QEnumComboBox, QIconifyIcon

SPIN_GIF = str(Path(__file__).parent / "spin.gif")


class DimToggleButton(QPushButton):
    def __init__(self, parent: QWidget | None = None):
        icn = QIconifyIcon("f7:view-2d", color="#333333")
        icn.addKey("f7:view-3d", state=QIconifyIcon.State.On, color="white")
        super().__init__(icn, "", parent)
        self.setCheckable(True)
        self.setChecked(True)


class QSpinner(QLabel):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        size = QSize(16, 16)
        mov = QMovie(SPIN_GIF, parent=self)
        self.setFixedSize(size)
        mov.setScaledSize(size)
        mov.setSpeed(150)
        mov.start()
        self.setMovie(mov)
        self.hide()


class ChannelMode(str, Enum):
    COMPOSITE = "composite"
    RGB = "rgb"
    MONO = "mono"

    def __str__(self) -> str:
        return self.value


class ChannelModeCombo(QEnumComboBox):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, enum_class=ChannelMode)

    def enable_rgb(self, enable: bool) -> None:
        current = self.currentEnum()
        self.setEnumClass(ChannelMode)
        if not enable:
            idx = list(ChannelMode.__members__.keys()).index("RGB")
            self.removeItem(idx)
        if current:
            self.setCurrentEnum(current)


class ROIButton(QPushButton):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setToolTip("Add ROI")
        self.setIcon(QIconifyIcon("mdi:vector-rectangle"))
