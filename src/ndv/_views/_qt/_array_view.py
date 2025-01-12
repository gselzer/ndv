from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible, QElidingLabel, QLabeledRangeSlider, QLabeledSlider
from superqt.cmap import QColormapComboBox
from superqt.iconify import QIconifyIcon
from superqt.utils import signals_blocked

from ndv._views.bases import ArrayView, LutView
from ndv.models._array_display_model import ChannelMode

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    import cmap
    from qtpy.QtGui import QIcon

    from ndv._types import AxisKey

SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 15px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(128, 128, 128, 0.25),
        stop:1 rgba(128, 128, 128, 0.1)
    );
    border-radius: 3px;
}

QSlider::handle:horizontal {
    width: 38px;
    background: #999999;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(100, 100, 100, 0.25),
        stop:1 rgba(100, 100, 100, 0.1)
    );
}

QLabel { font-size: 12px; }

QRangeSlider { qproperty-barColor: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(100, 80, 120, 0.2),
        stop:1 rgba(100, 80, 120, 0.4)
    )}
"""


class _CmapCombo(QColormapComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, allow_user_colormaps=True, add_colormap_text="Add...")
        self.setMinimumSize(140, 21)
        # self.setStyleSheet("background-color: transparent;")

    def showPopup(self) -> None:
        super().showPopup()
        popup = self.findChild(QFrame)
        popup.setMinimumWidth(self.width() + 100)
        popup.move(popup.x(), popup.y() - self.height() - popup.height())

    # TODO: upstream me
    def setCurrentColormap(self, cmap_: cmap.Colormap) -> None:
        """Adds the color to the QComboBox and selects it."""
        for idx in range(self.count()):
            if item := self.itemColormap(idx):
                if item.name == cmap_.name:
                    self.setCurrentIndex(idx)
        else:
            self.addColormap(cmap_)


class _QLUTWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.visible = QCheckBox()

        self.cmap = _CmapCombo()
        self.cmap.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cmap.addColormaps(["gray", "green", "magenta"])

        self.clims = QLabeledRangeSlider(Qt.Orientation.Horizontal)

        WHITE_SS = SLIDER_STYLE + "SliderLabel { font-size: 10px; color: white;}"
        self.clims.setStyleSheet(WHITE_SS)
        self.clims.setHandleLabelPosition(
            QLabeledRangeSlider.LabelPosition.LabelsOnHandle
        )
        self.clims.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        self.clims.setRange(0, 2**16)  # TODO: expose

        self.auto_clim = QPushButton("Auto")
        self.auto_clim.setMaximumWidth(42)
        self.auto_clim.setCheckable(True)

        layout = QHBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.visible)
        layout.addWidget(self.cmap)
        layout.addWidget(self.clims)
        layout.addWidget(self.auto_clim)


class QLutView(LutView):
    def __init__(self) -> None:
        super().__init__()
        self._qwidget = _QLUTWidget()
        # TODO: use emit_fast
        self._qwidget.visible.toggled.connect(self._on_q_visibility_changed)
        self._qwidget.cmap.currentColormapChanged.connect(self._on_q_cmap_changed)
        self._qwidget.clims.valueChanged.connect(self._on_q_clims_changed)
        self._qwidget.auto_clim.toggled.connect(self._on_q_auto_changed)

    def frontend_widget(self) -> QWidget:
        return self._qwidget

    def set_channel_name(self, name: str) -> None:
        self._qwidget.visible.setText(name)

    def set_auto_scale(self, auto: bool) -> None:
        self._qwidget.auto_clim.setChecked(auto)

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        self._qwidget.cmap.setCurrentColormap(cmap)

    def set_clims(self, clims: tuple[float, float]) -> None:
        self._qwidget.clims.setValue(clims)

    def set_gamma(self, gamma: float) -> None:
        pass

    def set_channel_visible(self, visible: bool) -> None:
        self._qwidget.visible.setChecked(visible)

    def set_visible(self, visible: bool) -> None:
        self._qwidget.setVisible(visible)

    def close(self) -> None:
        self._qwidget.close()

    def _on_q_visibility_changed(self, visible: bool) -> None:
        if self._model:
            self._model.visible = visible

    def _on_q_cmap_changed(self, cmap: cmap.Colormap) -> None:
        if self._model:
            self._model.cmap = cmap

    def _on_q_clims_changed(self, clims: tuple[float, float]) -> None:
        if self._model:
            self._model.clims = clims

    def _on_q_auto_changed(self, autoscale: bool) -> None:
        if self._model:
            self._model.autoscale = autoscale


class _QDimsSliders(QWidget):
    currentIndexChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sliders: dict[Hashable, QLabeledSlider] = {}
        self.setStyleSheet(SLIDER_STYLE)

        layout = QFormLayout(self)
        layout.setSpacing(2)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setContentsMargins(0, 0, 0, 0)

    def create_sliders(self, coords: Mapping[int, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        layout = cast("QFormLayout", self.layout())
        for axis, _coords in coords.items():
            sld = QLabeledSlider(Qt.Orientation.Horizontal)
            sld.valueChanged.connect(self.currentIndexChanged)
            if isinstance(_coords, range):
                sld.setRange(_coords.start, _coords.stop - 1)
                sld.setSingleStep(_coords.step)
            else:
                sld.setRange(0, len(_coords) - 1)
            layout.addRow(str(axis), sld)
            self._sliders[axis] = sld
        self.currentIndexChanged.emit()

    def hide_dimensions(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        layout = cast("QFormLayout", self.layout())
        for ax, slider in self._sliders.items():
            if ax in axes_to_hide:
                layout.setRowVisible(slider, False)
            elif show_remainder:
                layout.setRowVisible(slider, True)

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        """Return the current value of the sliders."""
        return {axis: slider.value() for axis, slider in self._sliders.items()}

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        """Set the current value of the sliders."""
        changed = False
        # only emit signal if the value actually changed
        # NOTE: this may be unnecessary, since usually the only thing calling
        # set_current_index is the controller, which already knows the value
        # however, we use this method directly in testing and it's nice to ensure.
        with signals_blocked(self):
            for axis, val in value.items():
                if isinstance(val, slice):
                    raise NotImplementedError("Slices are not supported yet")
                if slider := self._sliders.get(axis):
                    if slider.value() != val:
                        changed = True
                        slider.setValue(val)
                else:  # pragma: no cover
                    warnings.warn(f"Axis {axis} not found in sliders", stacklevel=2)
        if changed:
            self.currentIndexChanged.emit()


class _UpCollapsible(QCollapsible):
    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
        expandedIcon: QIcon | str | None = "▼",
        collapsedIcon: QIcon | str | None = "▲",
    ):
        super().__init__(title, parent, expandedIcon, collapsedIcon)
        # little hack to make the lut collapsible take up less space
        layout = cast("QVBoxLayout", self.layout())
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if (
            # look-before-leap on private attribute that may change
            hasattr(self, "_content") and (inner := self._content.layout()) is not None
        ):
            inner.setContentsMargins(0, 4, 0, 0)
            inner.setSpacing(0)

        self.setDuration(100)

        # this is a little hack to allow the buttons on the main view (below)
        # share the same row as the LUT toggle button
        layout.removeWidget(self._toggle_btn)
        self.btn_row = QHBoxLayout()
        self.btn_row.setContentsMargins(0, 0, 0, 0)
        self.btn_row.setSpacing(0)
        self.btn_row.addWidget(self._toggle_btn)
        self.btn_row.addStretch()
        layout.addLayout(self.btn_row)

    def setContent(self, content: QWidget) -> None:
        """Replace central widget (the widget that gets expanded/collapsed)."""
        self._content = content
        # this is different from upstream
        cast("QVBoxLayout", self.layout()).insertWidget(0, self._content)
        self._animation.setTargetObject(content)


# this is a PView ... but that would make a metaclass conflict
class _QArrayViewer(QWidget):
    def __init__(self, canvas_widget: QWidget, parent: QWidget | None = None):
        super().__init__(parent)

        self.dims_sliders = _QDimsSliders(self)

        # place to display dataset summary
        self.data_info_label = QElidingLabel("", parent=self)
        # place to display arbitrary text
        self.hover_info_label = QElidingLabel("", self)

        # the button that controls the display mode of the channels
        # not using QEnumComboBox because we want to exclude some values for now
        self.channel_mode_combo = QComboBox(self)
        self.channel_mode_combo.addItems(
            [ChannelMode.GRAYSCALE.value, ChannelMode.COMPOSITE.value]
        )

        # button to reset the zoom of the canvas
        # TODO: unify icons across all the view frontends in a new file
        set_range_icon = QIconifyIcon("fluent:full-screen-maximize-24-filled")
        self.set_range_btn = QPushButton(set_range_icon, "", self)

        # button to add a histogram
        add_histogram_icon = QIconifyIcon("foundation:graph-bar")
        self.histogram_btn = QPushButton(add_histogram_icon, "", self)

        self.luts = _UpCollapsible(
            "LUTs",
            parent=self,
            expandedIcon=QIconifyIcon("bi:chevron-up", color="#888888"),
            collapsedIcon=QIconifyIcon("bi:chevron-down", color="#888888"),
        )
        self._btn_layout = self.luts.btn_row
        self._btn_layout.setParent(None)
        self.luts.expand()

        self._btn_layout.addWidget(self.channel_mode_combo)
        # self._btns.addWidget(self._ndims_btn)
        self._btn_layout.addWidget(self.histogram_btn)
        self._btn_layout.addWidget(self.set_range_btn)
        # self._btns.addWidget(self._add_roi_btn)

        # above the canvas
        info_widget = QWidget()
        info = QHBoxLayout(info_widget)
        info.setContentsMargins(0, 0, 0, 2)
        info.setSpacing(0)
        info.addWidget(self.data_info_label)
        info_widget.setFixedHeight(16)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(2)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(info_widget)
        left_layout.addWidget(canvas_widget, 1)
        left_layout.addWidget(self.hover_info_label)
        left_layout.addWidget(self.dims_sliders)
        left_layout.addWidget(self.luts)
        left_layout.addLayout(self._btn_layout)

        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.splitter.addWidget(left)

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.splitter)


class QtArrayView(ArrayView):
    def __init__(self, canvas_widget: QWidget) -> None:
        self._qwidget = qwdg = _QArrayViewer(canvas_widget)
        qwdg.histogram_btn.clicked.connect(self._on_add_histogram_clicked)

        # TODO: use emit_fast
        qwdg.dims_sliders.currentIndexChanged.connect(self.currentIndexChanged.emit)
        qwdg.channel_mode_combo.currentTextChanged.connect(
            self._on_channel_mode_changed
        )
        qwdg.set_range_btn.clicked.connect(self.resetZoomClicked.emit)

    def add_lut_view(self) -> QLutView:
        view = QLutView()
        self._qwidget.luts.addWidget(view.frontend_widget())
        return view

    def remove_lut_view(self, view: LutView) -> None:
        self._qwidget.luts.removeWidget(cast("QLutView", view).frontend_widget())

    def _on_channel_mode_changed(self, text: str) -> None:
        self.channelModeChanged.emit(ChannelMode(text))

    def _on_add_histogram_clicked(self) -> None:
        splitter = self._qwidget.splitter
        if hasattr(self, "_hist"):
            if not (sizes := splitter.sizes())[-1]:
                splitter.setSizes([self._qwidget.height() - 100, 100])
            else:
                splitter.setSizes([sum(sizes), 0])
        else:
            self.histogramRequested.emit()

    def add_histogram(self, widget: QWidget) -> None:
        if hasattr(self, "_hist"):
            raise RuntimeError("Only one histogram can be added at a time")
        self._hist = widget
        self._qwidget.splitter.addWidget(widget)
        self._qwidget.splitter.setSizes([self._qwidget.height() - 100, 100])

    def remove_histogram(self, widget: QWidget) -> None:
        widget.setParent(None)
        widget.deleteLater()

    def create_sliders(self, coords: Mapping[int, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        self._qwidget.dims_sliders.create_sliders(coords)

    def hide_sliders(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        """Hide sliders based on visible axes."""
        self._qwidget.dims_sliders.hide_dimensions(axes_to_hide, show_remainder)

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        """Return the current value of the sliders."""
        return self._qwidget.dims_sliders.current_index()

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        """Set the current value of the sliders."""
        self._qwidget.dims_sliders.set_current_index(value)

    def set_data_info(self, text: str) -> None:
        """Set the data info text, above the canvas."""
        self._qwidget.data_info_label.setText(text)

    def set_hover_info(self, text: str) -> None:
        """Set the hover info text, below the canvas."""
        self._qwidget.hover_info_label.setText(text)

    def set_channel_mode(self, mode: ChannelMode) -> None:
        """Set the channel mode button text."""
        self._qwidget.channel_mode_combo.setCurrentText(mode.value)

    def set_visible(self, visible: bool) -> None:
        self._qwidget.setVisible(visible)

    def close(self) -> None:
        self._qwidget.close()

    def frontend_widget(self) -> QWidget:
        return self._qwidget
