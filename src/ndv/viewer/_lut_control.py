from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QWidget,
)
from superqt import QLabeledRangeSlider
from superqt.cmap import QColormapComboBox
from superqt.utils import signals_blocked

# TODO: Generalize, allow multiple backends
from vispy import scene

from ._dims_slider import SS

if TYPE_CHECKING:
    from collections.abc import Iterable

    import cmap
    from cmap import Colormap
    from vispy.scene import events

    from ._backends._protocols import PImageHandle


class CmapCombo(QColormapComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, allow_user_colormaps=True, add_colormap_text="Add...")
        self.setMinimumSize(120, 21)
        # self.setStyleSheet("background-color: transparent;")

    def showPopup(self) -> None:
        super().showPopup()
        popup = self.findChild(QFrame)
        popup.setMinimumWidth(self.width() + 100)
        popup.move(popup.x(), popup.y() - self.height() - popup.height())


class LutControl(QWidget):
    def __init__(
        self,
        name: str = "",
        handles: Iterable[PImageHandle] = (),
        parent: QWidget | None = None,
        cmaplist: Iterable[Any] = (),
        auto_clim: bool = True,
    ) -> None:
        super().__init__(parent)
        self._handles = handles
        self._name = name

        self._visible = QCheckBox(name)
        self._visible.setChecked(True)
        self._visible.toggled.connect(self._on_visible_changed)

        self._cmap = CmapCombo()
        self._cmap.currentColormapChanged.connect(self._on_cmap_changed)
        for handle in handles:
            self._cmap.addColormap(handle.cmap)
        for color in cmaplist:
            self._cmap.addColormap(color)

        self._clims = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self._clims.setStyleSheet(SS)
        self._clims.setHandleLabelPosition(
            QLabeledRangeSlider.LabelPosition.LabelsOnHandle
        )
        self._clims.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        self._clims.setRange(0, 2**8)
        self._clims.valueChanged.connect(self._on_clims_changed)

        self._auto_clim = QPushButton("Auto")
        self._auto_clim.setMaximumWidth(42)
        self._auto_clim.setCheckable(True)
        self._auto_clim.setChecked(auto_clim)
        self._auto_clim.toggled.connect(self.update_autoscale)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._visible)
        layout.addWidget(self._cmap)
        layout.addWidget(self._clims)
        layout.addWidget(self._auto_clim)

        self.update_autoscale()

    def _get_state(self) -> dict[str, Any]:
        return {
            "visible": self._visible.isChecked(),
            "cmap": self._cmap.currentColormap(),
            "clims": self._clims.value(),
            "auto_clim": self._auto_clim.isChecked(),
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        self._visible.setChecked(state["visible"])
        self._cmap.setCurrentColormap(state["cmap"])
        self._clims.setValue(state["clims"])
        self._auto_clim.setChecked(state["auto_clim"])

    def autoscaleChecked(self) -> bool:
        return cast("bool", self._auto_clim.isChecked())

    def _on_clims_changed(self, clims: tuple[float, float]) -> None:
        self._auto_clim.setChecked(False)
        for handle in self._handles:
            handle.clim = clims

    def _on_visible_changed(self, visible: bool) -> None:
        for handle in self._handles:
            handle.visible = visible
        if visible:
            self.update_autoscale()

    def _on_cmap_changed(self, cmap: cmap.Colormap) -> None:
        for handle in self._handles:
            handle.cmap = cmap

    def update_autoscale(self) -> None:
        if (
            not self._auto_clim.isChecked()
            or not self._visible.isChecked()
            or not self._handles
        ):
            return

        # find the min and max values for the current channel
        clims = [np.inf, -np.inf]
        for handle in self._handles:
            clims[0] = min(clims[0], np.nanmin(handle.data))
            clims[1] = max(clims[1], np.nanmax(handle.data))

        mi, ma = tuple(int(x) for x in clims)
        for handle in self._handles:
            handle.clim = (mi, ma)

        # set the slider values to the new clims
        with signals_blocked(self._clims):
            self._clims.setMinimum(min(mi, self._clims.minimum()))
            self._clims.setMaximum(max(ma, self._clims.maximum()))
            self._clims.setValue((mi, ma))


class HistogramLutControl(LutControl):
    def __init__(
        self,
        name: str = "",
        handles: list[PImageHandle] | None = None,
        parent: QWidget | None = None,
        cmaplist: Iterable[Any] = (),
        auto_clim: bool = True,
    ) -> None:
        if handles is None:
            handles = []
        self._histogram = HistogramWidget()
        super().__init__(name, handles, parent, cmaplist, auto_clim)
        self._clims.setVisible(False)
        layout = self.layout()
        if not isinstance(layout, QHBoxLayout):
            raise Exception(f"LutControl layout is a {type(layout)}")
        layout.insertWidget(2, self._histogram, 1)

        if len(handles) == 1:
            self._histogram.dataChanged.emit(handles[0].data)

    def _on_cmap_changed(self, cmap: Colormap) -> None:
        super()._on_cmap_changed(cmap)
        self._histogram.recolor(cmap.color_stops[-1].color)

    def update_autoscale(self) -> None:
        # TODO: Handle (heh) multiple handles
        if (handle := next(iter(self._handles), None)) is not None:
            self._histogram.dataChanged.emit(handle.data)
            if not self._auto_clim.isChecked() or not self._visible.isChecked():
                return

            # find the min and max values for the current channel
            clims = [np.inf, -np.inf]
            for handle in self._handles:
                clims[0] = min(clims[0], np.nanmin(handle.data))
                clims[1] = max(clims[1], np.nanmax(handle.data))

            mi, ma = tuple(int(x) for x in clims)
            for handle in self._handles:
                handle.clim = (mi, ma)

            # set the slider values to the new clims
            with signals_blocked(self._histogram):
                self._histogram.min_box.setValue(mi)
                self._histogram._canvas.l._l_bound.value = mi

                self._histogram.max_box.setValue(ma)
                self._histogram._canvas.l._r_bound.value = ma

            # HACK
            self._auto_clim.setChecked(True)


class HistogramWidget(QWidget):
    dataChanged = Signal(np.ndarray)
    climsChanged = Signal(float, float)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        self.bgcolor = self.palette().color(QPalette.ColorRole.Base).getRgb()
        self._dtype: np.dtype = np.dtype("uint8")

        self.min_box = QDoubleSpinBox()
        self.min_box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.min_box.valueChanged.connect(self._on_min_box_changed)
        self.climsChanged.connect(self._on_clims_changed)
        self._canvas = HistogramCanvas(self, bgcolor=self.bgcolor)
        self.max_box = QDoubleSpinBox()
        self.max_box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.max_box.valueChanged.connect(self._on_max_box_changed)

        layout.addWidget(self.min_box)
        layout.addWidget(self._canvas.native)
        layout.addWidget(self.max_box)

        self.dataChanged.connect(self._on_data_change)

    def _on_min_box_changed(self, val: float) -> None:
        self._on_clims_changed(val, self.max_box.value())

    def _on_max_box_changed(self, val: float) -> None:
        self._on_clims_changed(self.min_box.value(), val)

    def recolor(self, color: Any) -> None:
        self._canvas.l._mesh.color = color

    def _on_clims_changed(self, cmin: float, cmax: float) -> None:
        with signals_blocked(self):
            self.min_box.setValue(cmin)
            self._canvas.l._l_bound.value = cmin

            self.max_box.setValue(cmax)
            self._canvas.l._r_bound.value = cmax
        if (parent := self.parent()) is not None:
            parent._on_clims_changed((cmin, cmax))

    def _on_data_change(self, data: np.ndarray) -> None:
        if data.dtype != self._dtype:
            self._dtype = data.dtype
            try:
                info: np.iinfo | np.finfo = np.iinfo(self._dtype)
            except Exception:
                # NB: don't know how to find out type info if this one also fails
                info = np.finfo(self._dtype)

            self._canvas._range = (info.min, info.max)
            r = self._canvas._range[1] - self._canvas._range[0]
            self._canvas._view.camera.rect = (
                self._canvas._range[0] - (r * 0.1),  # x pos of lower left corner
                -0.1,  # y pos of lower left corner
                r * 1.2,  # width
                1.2,  # height
            )
            self._canvas.l._l_bound.transform.scale = [r * 0.05, 1]
            self._canvas.l._r_bound.transform.scale = [r * 0.05, 1]
            self.min_box.setRange(*self._canvas._range)
            self.max_box.setRange(*self._canvas._range)
            self._on_clims_changed(*self._canvas._range)

        self._canvas.l._mesh.data = data


class Bound(scene.visuals.Polygon):
    def __init__(self, x: float, left: bool, *args: Any, **kwargs: Any) -> None:
        pos: list[list[float]] = [[0, 0], [1, 0.5], [0, 1]]
        if left:
            pos[1][0] = -1
        kwargs["pos"] = pos

        kwargs.setdefault("color", "white")
        kwargs.setdefault("border_color", "black")

        scene.visuals.Polygon.__init__(
            self,
            *args,
            **kwargs,
        )

        self.unfreeze()
        self.interactive = True
        self.transform = scene.transforms.STTransform()

        self._move_offset = [0, 0, 0, 0]

        self.freeze()

    def start_move(self, pos: tuple[float, float]) -> None:
        self._move_offset[:2] = pos - self.transform.translate[:2]

    def move(self, pos: tuple[float, float]) -> None:
        t = self.transform.translate
        t[0] = max(0, min(1, pos[0] - self._move_offset[0]))
        self.transform.translate = t

    @property
    def value(self) -> float:
        return float(self.transform.translate[0])

    @value.setter
    def value(self, v: float) -> None:
        t = self.transform.translate
        t[0] = v
        self.transform.translate = t


# TODO: Create Visual?
class LutEditor(scene.visuals.Compound):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__([], *args, **kwargs)

        self.unfreeze()

        self._l_bound = Bound(parent=self, x=0, left=True)
        self._r_bound = Bound(parent=self, x=1, left=False)
        self._mesh = Histogram(parent=self)

        self.freeze()


class Histogram(scene.visuals.Mesh):
    def __init__(
        self,
        data: np.ndarray | None = None,
        color: Any = "white",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs["color"] = color
        super().__init__(*args, **kwargs)

        self.unfreeze()
        # the last future that was created by _update_data_for_index
        self._last_future: Future | None = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.bins = 256
        self.range: tuple[float, float] = (0, 1)
        self._data: np.ndarray = np.empty(())
        if data is not None:
            self.data = data

        self.freeze()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, d: np.ndarray) -> None:
        self._data = d
        try:
            iinfo = np.iinfo(self._data.dtype)
            self.range = (iinfo.min, iinfo.max)
        except Exception as exc:
            raise Exception(f"Unsupported dtype: {self._data.dtype}") from exc
        if self._last_future:
            self._last_future.cancel()
        self._last_future = f = self._executor.submit(
            lambda: self._recompute_vertices(d)
        )
        f.add_done_callback(self._update_histogram)

    def _recompute_vertices(
        self, bin_vals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Graciously adapted from vispy.visuals.histogram.py."""
        #   4-5
        #   | |
        # 1-2/7-8
        # |/| | |
        # 0-3-6-9
        #

        # do the histogramming
        bin_vals, bin_edges = np.histogram(bin_vals, self.bins, range=self.range)
        # normalize histograms
        bin_vals = bin_vals / np.max(bin_vals)
        # construct our vertices
        rr = np.zeros((3 * len(bin_edges) - 2, 3), np.float32)
        rr[:, 0] = np.repeat(bin_edges, 3)[1:-1]
        rr[1::3, 1] = bin_vals
        rr[2::3, 1] = bin_vals
        bin_edges.astype(np.float32)
        # and now our tris
        tris = np.zeros((2 * len(bin_edges) - 2, 3), np.uint32)
        offsets = 3 * np.arange(len(bin_edges) - 1, dtype=np.uint32)[:, np.newaxis]
        tri_1 = np.array([0, 2, 1])
        tri_2 = np.array([2, 0, 3])
        tris[::2] = tri_1 + offsets
        tris[1::2] = tri_2 + offsets

        return (rr, tris)

    def _update_histogram(self, future: Future[tuple[np.ndarray, np.ndarray]]) -> None:
        # NOTE: removing the reference to the last future here is important
        # because the future has a reference to this widget in its _done_callbacks
        # which will prevent the widget from being garbage collected if the future
        self._last_future = None
        if future.cancelled():
            return
        data = future.result()
        self.set_data(vertices=data[0], faces=data[1])


class HistogramCanvas(scene.SceneCanvas):
    def __init__(self, wdg: HistogramWidget, bgcolor: Any = "black") -> None:
        super().__init__(
            show=True,
            # size=(200, 100),
        )

        self.unfreeze()
        central_wdg: scene.Widget = self.central_widget
        self._view: scene.ViewBox = central_wdg.add_view()
        if isinstance(bgcolor, Sequence) and not isinstance(bgcolor, str):
            bgcolor = [float(x) / 255 for x in bgcolor]
        self._view.bgcolor = bgcolor

        self.l = LutEditor(
            parent=self._view.scene,
        )
        self._view.camera = self._camera = scene.PanZoomCamera(rect=(0, 0, 0, 0))
        self.l.transform = scene.transforms.STTransform()
        # self._view.camera = self._camera
        self._camera.interactive = False
        self._wdg = wdg
        self._on_mouse_move: list[Callable[[tuple[int, int]], None]] = []
        self.pressed: Bound | None = None
        self.press_offset: float = 0
        self._range = (0, 1)

        self.freeze()

    def on_mouse_press(self, event: events.SceneMouseEvent) -> None:
        pos = self._view.scene.transform.imap(event.pos)[:2]
        pos = self.l.transform.imap(pos)[:2]
        for v in self.visuals_at(event.pos):
            if isinstance(v, Bound):
                self.pressed = v
                self.press_offset = pos[0] - v.value
                return
        self.pressed = None

    def on_mouse_move(self, event: events.SceneMouseEvent) -> None:
        if self.pressed is None:
            return
        pos = self._view.scene.transform.imap(event.pos)[:2]
        pos = self.l.transform.imap(pos)[0] - self.press_offset
        if self.pressed is self.l._l_bound:
            bound_pos = max(self._range[0], min(pos, self.l._r_bound.value))
            self.l._l_bound.value = bound_pos
            self._wdg.climsChanged.emit(self.l._l_bound.value, self.l._r_bound.value)
        elif self.pressed is self.l._r_bound:
            bound_pos = max(self.l._l_bound.value, min(pos, self._range[1]))
            self.l._r_bound.value = bound_pos
            self._wdg.climsChanged.emit(self.l._l_bound.value, self.l._r_bound.value)

    def on_mouse_release(self, event: events.SceneMouseEvent) -> None:
        self.pressed = None
