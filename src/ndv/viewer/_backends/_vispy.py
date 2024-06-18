from __future__ import annotations

import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

import cmap
import numpy as np
import vispy
import vispy.scene
import vispy.visuals
from qtpy.QtCore import Qt
from vispy import scene
from vispy.color import Color
from vispy.util.quaternion import Quaternion

from ._protocols import CanvasMode

if TYPE_CHECKING:
    from typing import Callable

    from qtpy.QtWidgets import QWidget
    from vispy.scene.events import SceneMouseEvent

turn = np.sin(np.pi / 4)
DEFAULT_QUATERNION = Quaternion(turn, turn, 0, 0)


class ROIElement:
    def select(self, visible: bool) -> None:
        raise NotImplementedError("Must be implemented in subclass")

    def start_move(self, event: SceneMouseEvent) -> None:
        raise NotImplementedError("Must be implemented in subclass")

    def move(self, pos: Sequence[float]) -> None:
        raise NotImplementedError("Must be implemented in subclass")


class Handle(scene.visuals.Markers, ROIElement):
    """A draggable Marker that is part of a ROI."""

    def __init__(
        self, parent: EditableROI, on_move: Callable[[list[float]], None] | None = None
    ) -> None:
        super().__init__(parent=parent)
        self.unfreeze()
        self.parent = parent
        # Callback function(s)
        self.on_move: list[Callable[[list[float]], None]] = []
        if on_move:
            self.on_move.append(on_move)

        # NB VisPy asks that the data is a 2D array
        self._pos = np.array([[0, 0]], dtype=np.float32)
        self.interactive = True
        self.freeze()

    def start_move(self, event: SceneMouseEvent) -> None:
        pass

    def move(self, pos: list[float]) -> None:
        for func in self.on_move:
            func(pos)

    @property
    def pos(self) -> list[float]:
        return cast(list[float], self._pos[0, :])

    @pos.setter
    def pos(self, pos: list[float]) -> None:
        self._pos[:] = pos
        self.set_data(self._pos)

    def select(self, visible: bool) -> None:
        self.parent.select(visible)


class EditableROI(scene.visuals.Polygon, ROIElement):
    def vertices(self) -> list[list[float]]:
        raise NotImplementedError("Must be implemented in subclass")

    def select(self, visible: bool) -> None:
        raise NotImplementedError("Must be implemented in subclass")


class RectangularROI(scene.visuals.Rectangle, EditableROI):
    def __init__(
        self,
        parent: scene.visuals.Visual,
        center: list[float] | None = None,
        width: float = 1e-6,
        height: float = 1e-6,
    ) -> None:
        if center is None:
            center = [0, 0]
        scene.visuals.Rectangle.__init__(
            self, center=center, width=width, height=height, radius=0, parent=parent
        )
        self.unfreeze()
        self.parent = parent
        self.interactive = True

        self._handles = [
            Handle(self, on_move=self.move_top_left),
            Handle(self, on_move=self.move_top_right),
            Handle(self, on_move=self.move_bottom_right),
            Handle(self, on_move=self.move_bottom_left),
        ]

        # drag_reference defines the offset between where the user clicks and the center
        # of the rectangle
        self.drag_reference = [0.0, 0.0]
        self.interactive = True
        self.freeze()

    def move_top_left(self, pos: list[float]) -> None:
        self._handles[3].pos = [pos[0], self._handles[3].pos[1]]
        self._handles[0].pos = pos
        self._handles[1].pos = [self._handles[1].pos[0], pos[1]]
        self.redraw()

    def move_top_right(self, pos: list[float]) -> None:
        self._handles[0].pos = [self._handles[0].pos[0], pos[1]]
        self._handles[1].pos = pos
        self._handles[2].pos = [pos[0], self._handles[2].pos[1]]
        self.redraw()

    def move_bottom_right(self, pos: list[float]) -> None:
        self._handles[1].pos = [pos[0], self._handles[1].pos[1]]
        self._handles[2].pos = pos
        self._handles[3].pos = [self._handles[3].pos[0], pos[1]]
        self.redraw()

    def move_bottom_left(self, pos: list[float]) -> None:
        self._handles[2].pos = [self._handles[2].pos[0], pos[1]]
        self._handles[3].pos = pos
        self._handles[0].pos = [pos[0], self._handles[0].pos[1]]
        self.redraw()

    def start_move(self, pos: list[float]) -> None:
        self.drag_reference = [
            pos[0] - self.center[0],
            pos[1] - self.center[1],
        ]

    def redraw(self) -> None:
        left, top, *_ = self._handles[0].pos
        right, bottom, *_ = self._handles[2].pos

        self.center = [(left + right) / 2, (top + bottom) / 2]
        self.width = max(abs(left - right), 1e-6)
        self.height = max(abs(top - bottom), 1e-6)

    def move(self, pos: list[float]) -> None:
        new_center = [
            pos[0] - self.drag_reference[0],
            pos[1] - self.drag_reference[1],
        ]
        old_center = self.center
        for h in self._handles:
            h.pos += [new_center[0] - old_center[0], new_center[1] - old_center[1]]
        self.center = new_center

    def select(self, visible: bool) -> None:
        for h in self._handles:
            h.visible = visible


class VispyImageHandle:
    def __init__(self, visual: scene.visuals.Image | scene.visuals.Volume) -> None:
        self._visual = visual
        self._ndim = 2 if isinstance(visual, scene.visuals.Image) else 3

    @property
    def data(self) -> np.ndarray:
        try:
            return self._visual._data  # type: ignore [no-any-return]
        except AttributeError:
            return self._visual._last_data  # type: ignore [no-any-return]

    @data.setter
    def data(self, data: np.ndarray) -> None:
        if not data.ndim == self._ndim:
            warnings.warn(
                f"Got wrong number of dimensions ({data.ndim}) for vispy "
                f"visual of type {type(self._visual)}.",
                stacklevel=2,
            )
            return
        self._visual.set_data(data)

    @property
    def visible(self) -> bool:
        return bool(self._visual.visible)

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._visual.visible = visible

    @property
    def clim(self) -> Any:
        return self._visual.clim

    @clim.setter
    def clim(self, clims: tuple[float, float]) -> None:
        with suppress(ZeroDivisionError):
            self._visual.clim = clims

    @property
    def cmap(self) -> cmap.Colormap:
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: cmap.Colormap) -> None:
        self._cmap = cmap
        self._visual.cmap = cmap.to_vispy()

    @property
    def transform(self) -> np.ndarray:
        raise NotImplementedError

    @transform.setter
    def transform(self, transform: np.ndarray) -> None:
        raise NotImplementedError

    def remove(self) -> None:
        self._visual.parent = None


class VispyRoiHandle:
    def __init__(self, roi: RectangularROI) -> None:
        self._roi = roi

    @property
    def vertices(self) -> list[tuple[float, float]]:
        cp = self._roi.control_points
        # TODO - is there any way for us to get the position from each Marker?
        return [
            (cp._left, cp._bottom),
            (cp._right, cp._bottom),
            (cp._right, cp._top),
            (cp._left, cp._top),
        ]

    @vertices.setter
    def vertices(self, data: list[tuple[float, float]]) -> None:
        if len(data) != 4:
            raise Exception(
                "Only rectangles aligned with the axes are currently supported"
            )
        is_aligned = (
            data[0][1] == data[1][1]
            and data[1][0] == data[2][0]
            and data[2][1] == data[3][1]
            and data[3][0] == data[0][0]
        )
        if not is_aligned:
            raise Exception(
                "Only rectangles aligned with the axes are currently supported"
            )

        cp = self._roi.control_points
        cp._left = data[0][0]
        cp._right = data[1][0]
        cp._bottom = data[1][1]
        cp._top = data[2][1]
        cp.update_points()
        self._roi.update_from_controlpoints()

    @property
    def visible(self) -> bool:
        return bool(self._roi.visible)

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._roi.visible = visible

    @property
    def color(self) -> Any:
        return self._roi.color

    @color.setter
    def color(self, color: cmap.Color | None = None) -> None:
        if color is None:
            color = cmap.Color("transparent")
        # NB: To enable dragging the shape within the border,
        # we require a positive alpha.
        alpha = max(color.alpha, 1e-6)
        self._roi.color = Color(color.hex, alpha=alpha)

    @property
    def border_color(self) -> Any:
        return self._roi.border_color

    @border_color.setter
    def border_color(self, color: cmap.Color | None = None) -> None:
        if color is None:
            color = cmap.Color("yellow")
        self._roi.border_color = Color(color.hex, alpha=color.alpha)

    def remove(self) -> None:
        self._roi.parent = None

    def cursor_at(self, pos_xy: tuple[float, float]) -> Qt.CursorShape | None:
        cx, cy = self._roi.center
        w = self._roi.width / 2
        h = self._roi.height / 2
        threshold = 2
        h_dist = cx - pos_xy[0]  # horiz distance from the center
        v_dist = cy - pos_xy[1]  # vert  distance from the center

        # first check if we're outside the rectangle
        # (including a threshold to allow for easier dragging on the handles)
        if abs(h_dist) > (w + threshold) or abs(v_dist) > (h + threshold):
            return None

        # check if we're on a corner
        on_top_edge = h - v_dist < threshold
        on_bottom_edge = h + v_dist < threshold
        if w - h_dist < threshold:  # on the left edge
            if on_top_edge:
                return Qt.CursorShape.SizeFDiagCursor
            if on_bottom_edge:
                return Qt.CursorShape.SizeBDiagCursor
        if w + h_dist < threshold:  # on the right edge
            if on_top_edge:
                return Qt.CursorShape.SizeBDiagCursor
            if on_bottom_edge:
                return Qt.CursorShape.SizeFDiagCursor

        # now check if we're strictly outside the rectangle
        if abs(h_dist) > w or abs(v_dist) > h:
            return None

        # were inside the rectangle
        return Qt.CursorShape.SizeAllCursor


class VispyViewerCanvas:
    """Vispy-based viewer for data.

    All vispy-specific code is encapsulated in this class (and non-vispy canvases
    could be swapped in if needed as long as they implement the same interface).
    """

    def __init__(self) -> None:
        self._mode = CanvasMode.PAN_ZOOM
        self._canvas = scene.SceneCanvas()
        self._canvas.events.mouse_press.connect(self._on_mouse_press)
        self._current_shape: tuple[int, ...] = ()
        self._last_state: dict[Literal[2, 3], Any] = {}

        central_wdg: scene.Widget = self._canvas.central_widget
        self._view: scene.ViewBox = central_wdg.add_view()
        self._ndim: Literal[2, 3] | None = None

        self._selection: ROIElement | None = None

    @property
    def _camera(self) -> vispy.scene.cameras.BaseCamera:
        return self._view.camera

    def set_ndim(self, ndim: Literal[2, 3]) -> None:
        """Set the number of dimensions of the displayed data."""
        if ndim == self._ndim:
            return
        elif self._ndim is not None:
            # remember the current state before switching to the new camera
            self._last_state[self._ndim] = self._camera.get_state()

        self._ndim = ndim
        if ndim == 3:
            cam = scene.ArcballCamera(fov=0)
            # this sets the initial view similar to what the panzoom view would have.
            cam._quaternion = DEFAULT_QUATERNION
        else:
            cam = scene.PanZoomCamera(aspect=1, flip=(0, 1))

        # restore the previous state if it exists
        if state := self._last_state.get(ndim):
            cam.set_state(state)
        self._view.camera = cam

    def qwidget(self) -> QWidget:
        return cast("QWidget", self._canvas.native)

    def refresh(self) -> None:
        self._canvas.update()

    def add_image(
        self, data: np.ndarray | None = None, cmap: cmap.Colormap | None = None
    ) -> VispyImageHandle:
        """Add a new Image node to the scene."""
        img = scene.visuals.Image(data, parent=self._view.scene)
        img.set_gl_state("additive", depth_test=False)
        img.interactive = True
        if data is not None:
            self._current_shape, prev_shape = data.shape, self._current_shape
            if not prev_shape:
                self.set_range()
        handle = VispyImageHandle(img)
        if cmap is not None:
            handle.cmap = cmap
        return handle

    def add_volume(
        self, data: np.ndarray | None = None, cmap: cmap.Colormap | None = None
    ) -> VispyImageHandle:
        vol = scene.visuals.Volume(
            data, parent=self._view.scene, interpolation="nearest"
        )
        vol.set_gl_state("additive", depth_test=False)
        vol.interactive = True
        if data is not None:
            self._current_shape, prev_shape = data.shape, self._current_shape
            if len(prev_shape) != 3:
                self.set_range()
        handle = VispyImageHandle(vol)
        if cmap is not None:
            handle.cmap = cmap
        return handle

    def add_roi(
        self,
        vertices: list[tuple[float, float]] | None = None,
        color: Any | None = None,
        border_color: Any | None = None,
    ) -> VispyRoiHandle:
        """Add a new Rectangular ROI node to the scene."""
        roi = RectangularROI(
            parent=self._view.scene,
        )
        # Start by selecting the bottom right handle
        self._selection = roi._handles[2]

        handle = VispyRoiHandle(roi)
        if vertices:
            handle.vertices = vertices
        handle.color = color
        handle.border_color = border_color
        return handle

    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = 0.01,
    ) -> None:
        """Update the range of the PanZoomCamera.

        When called with no arguments, the range is set to the full extent of the data.
        """
        if len(self._current_shape) >= 2:
            if x is None:
                x = (0, self._current_shape[-1])
            if y is None:
                y = (0, self._current_shape[-2])
        if z is None and len(self._current_shape) == 3:
            z = (0, self._current_shape[-3])
        is_3d = isinstance(self._camera, scene.ArcballCamera)
        if is_3d:
            self._camera._quaternion = DEFAULT_QUATERNION
        self._view.camera.set_range(x=x, y=y, z=z, margin=margin)
        if is_3d:
            max_size = max(self._current_shape)
            self._camera.scale_factor = max_size + 6

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        return self._view.camera.transform.imap(pos_xy)[:3]  # type: ignore [no-any-return]

    def set_mode(self, mode: CanvasMode) -> None:
        self._mode = mode
        if self._mode is CanvasMode.EDIT_ROI:
            self._camera.interactive = False
        elif self._mode is CanvasMode.PAN_ZOOM:
            self._camera.interactive = True

    def _on_mouse_press(self, event: SceneMouseEvent) -> None:
        pos = self.canvas_to_world(event.pos)[:2]
        if self._mode == CanvasMode.EDIT_ROI and isinstance(self._selection, Handle):
            self._selection.parent.move(pos)
        else:
            selected = self._canvas.visual_at(event.pos)
            if isinstance(selected, ROIElement):
                self._selection = selected
            elif self._selection is not None:
                self._selection.select(visible=False)
                self._selection = None

        if self._selection is not None:
            self._selection.select(visible=True)
            self._selection.start_move(pos)
            self._camera.interactive = False
            handler = self._on_mouse_move_selection(self._selection)
            self._canvas.events.mouse_move.connect(handler)

            def disconnect(event: SceneMouseEvent) -> None:
                self._canvas.events.mouse_move.disconnect(handler)
                self._canvas.events.mouse_release.disconnect(disconnect)
                self._camera.interactive = True

            self._canvas.events.mouse_release.connect(disconnect)

    def _on_mouse_move_selection(
        self, selection: ROIElement
    ) -> Callable[[SceneMouseEvent], None]:
        def fooooooo(event: SceneMouseEvent) -> None:
            selection.move(self.canvas_to_world(event.pos)[:2])

        return fooooooo
