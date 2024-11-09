from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    import cmap
    import numpy as np
    from psygnal import Signal, SignalInstance
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QWidget

    from ndv._types import AxisKey


class PLutView(Protocol):
    visibleChanged: Signal
    autoscaleChanged: Signal
    cmapChanged: Signal
    climsChanged: Signal

    def setName(self, name: str) -> None: ...
    def setAutoScale(self, auto: bool) -> None: ...
    def setColormap(self, cmap: cmap.Colormap) -> None: ...
    def setClims(self, clims: tuple[float, float]) -> None: ...
    def setLutVisible(self, visible: bool) -> None: ...


class CanvasElement(Protocol):
    """Protocol defining an interactive element on the Canvas."""

    @property
    def visible(self) -> bool:
        """Defines whether the element is visible on the canvas."""

    @visible.setter
    def visible(self, visible: bool) -> None:
        """Sets element visibility."""

    @property
    def can_select(self) -> bool:
        """Defines whether the element can be selected."""

    @property
    def selected(self) -> bool:
        """Returns element selection status."""

    @selected.setter
    def selected(self, selected: bool) -> None:
        """Sets element selection status."""

    def cursor_at(self, pos: Sequence[float]) -> Qt.CursorShape | None:
        """Returns the element's cursor preference at the provided position."""

    def start_move(self, pos: Sequence[float]) -> None:
        """
        Behavior executed at the beginning of a "move" operation.

        In layman's terms, this is the behavior executed during the the "click"
        of a "click-and-drag".
        """

    def move(self, pos: Sequence[float]) -> None:
        """
        Behavior executed throughout a "move" operation.

        In layman's terms, this is the behavior executed during the "drag"
        of a "click-and-drag".
        """

    def remove(self) -> None:
        """Removes the element from the canvas."""


class PImageHandle(CanvasElement, Protocol):
    @property
    def data(self) -> np.ndarray: ...
    @data.setter
    def data(self, data: np.ndarray) -> None: ...
    @property
    def clim(self) -> Any: ...
    @clim.setter
    def clim(self, clims: tuple[float, float]) -> None: ...
    @property
    def cmap(self) -> cmap.Colormap: ...
    @cmap.setter
    def cmap(self, cmap: cmap.Colormap) -> None: ...


class PView(Protocol):
    """Protocol that front-end viewers must implement."""

    currentIndexChanged: SignalInstance

    def refresh(self) -> None: ...
    def create_sliders(self, coords: Mapping[int, Sequence]) -> None: ...
    def current_index(self) -> Mapping[AxisKey, int | slice]: ...
    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None: ...
    def add_image_to_canvas(self, data: Any) -> PImageHandle: ...
    def hide_sliders(
        self, axes_to_hide: Container[Hashable], *, show_remainder: bool = ...
    ) -> None: ...

    def add_lut_view(self) -> PLutView: ...
    def show(self) -> None: ...


class PRoiHandle(CanvasElement, Protocol):
    @property
    def vertices(self) -> Sequence[Sequence[float]]: ...
    @vertices.setter
    def vertices(self, data: Sequence[Sequence[float]]) -> None: ...
    @property
    def color(self) -> Any: ...
    @color.setter
    def color(self, color: cmap.Color) -> None: ...
    @property
    def border_color(self) -> Any: ...
    @border_color.setter
    def border_color(self, color: cmap.Color) -> None: ...


class PCanvas(Protocol):
    def __init__(self) -> None: ...
    def set_ndim(self, ndim: Literal[2, 3]) -> None: ...
    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = ...,
    ) -> None: ...
    def refresh(self) -> None: ...
    def qwidget(self) -> QWidget: ...
    def add_image(
        self, data: np.ndarray | None = ..., cmap: cmap.Colormap | None = ...
    ) -> PImageHandle: ...
    def add_volume(
        self, data: np.ndarray | None = ..., cmap: cmap.Colormap | None = ...
    ) -> PImageHandle: ...
    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""

    def elements_at(self, pos_xy: tuple[float, float]) -> list[CanvasElement]: ...
    def add_roi(
        self,
        vertices: Sequence[tuple[float, float]] | None = None,
        color: cmap.Color | None = None,
        border_color: cmap.Color | None = None,
    ) -> PRoiHandle: ...
