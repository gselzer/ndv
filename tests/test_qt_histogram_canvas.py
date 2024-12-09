from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QWidget

from ndv.models._stats import Stats
from ndv.views._qt.qt_view import QtHistogramView
from ndv.views._vispy._vispy import VispyHistogramCanvas

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

# Accounts for differences between 32-bit and 64-bit floats
EPSILON = 1e-6


@pytest.fixture
def stats() -> Stats:
    gen = np.random.default_rng(seed=0xDEADBEEF)
    data = gen.normal(10, 10, 10000).astype(np.float64)
    return Stats(data)


def test_log_btn(qtbot: QtBot, stats: Stats) -> None:
    _hist = VispyHistogramCanvas()
    _hist.set_stats(stats)
    view = QtHistogramView(_hist)
    qtbot.addWidget(view)
    # Assert log button deselected by default
    assert view._log.isCheckable()
    assert not view._log.isChecked()

    # Toggle, assert logarithmic data
    original_range = _hist.plot.yaxis.axis.domain[1]
    view._log.toggle()
    log_range = _hist.plot.yaxis.axis.domain[1]
    assert abs(math.log10(original_range) - log_range) <= EPSILON

    # Toggle again, assert logarithmic data
    view._log.toggle()
    linear_range = _hist.plot.yaxis.axis.domain[1]
    assert abs(original_range - linear_range) <= EPSILON


def test_resize_btn(qtbot: QtBot, stats: Stats) -> None:
    _hist = VispyHistogramCanvas()
    _hist.set_stats(stats)
    view = QtHistogramView(_hist)
    qtbot.addWidget(view)

    resizer = view._set_range_btn
    original_domain = _hist.plot.xaxis.axis.domain

    # Pan one "frame" right
    _hist.set_domain((original_domain[1], original_domain[1] + 1))
    new_domain = _hist.plot.xaxis.axis.domain
    assert abs(new_domain[0] - original_domain[1]) <= EPSILON
    assert abs(new_domain[1] - (original_domain[1] + 1)) <= EPSILON

    # Reset zoom
    resizer.click()
    reset_domain = _hist.plot.xaxis.axis.domain
    assert abs(reset_domain[0] - original_domain[0]) <= EPSILON
    assert abs(reset_domain[1] - original_domain[1]) <= EPSILON


def test_set_cursor(qtbot: QtBot, stats: Stats) -> None:
    _hist = VispyHistogramCanvas()
    _hist.set_stats(stats)
    view = QtHistogramView(_hist)
    qtbot.addWidget(view)
    qwdg = cast(QWidget, _hist.widget())

    _hist.set_lut_visible(True)
    _hist.set_domain((0, 100))
    _hist.set_clims((10, 90))

    def move_mouse(scene_pos: tuple[float, float]) -> None:
        canvas_pos = _hist.node_tform.imap(scene_pos)[:2]
        mouse_event = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(*canvas_pos),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        view.eventFilter(qwdg, mouse_event)

    # Test left clim
    move_mouse((10, 0))
    assert qwdg.cursor().shape() == Qt.CursorShape.SplitHCursor

    # Test gamma
    move_mouse(_hist._handle_transform.map([50, 0.5])[:2])
    assert qwdg.cursor().shape() == Qt.CursorShape.SplitVCursor

    # Test moving over nothing (grabbable)
    move_mouse((30, 0))
    assert qwdg.cursor().shape() == Qt.CursorShape.SizeAllCursor
