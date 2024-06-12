from __future__ import annotations

import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import cmap
import numpy as np
import vispy
import vispy.scene
import vispy.visuals
from qtpy.QtCore import Qt
from superqt.utils import qthrottled
from vispy import scene
from vispy.color import Color
from vispy.util.quaternion import Quaternion

from ._protocols import CanvasMode

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget
    from vispy.scene.events import SceneMouseEvent

turn = np.sin(np.pi / 4)
DEFAULT_QUATERNION = Quaternion(turn, turn, 0, 0)


class ControlPoints(scene.visuals.Compound):
    def __init__(self, parent: scene.visuals.Visual) -> None:
        scene.visuals.Compound.__init__(self, [])
        self.unfreeze()
        self.parent = parent
        self._top = 0.0
        self._bottom = 0.0
        self._left = 0.0
        self._right = 0.0
        self.selected_idx: int | None = None
        self.opposed_idx: int | None = None

        self.control_points = [scene.visuals.Markers(parent=self) for i in range(0, 4)]
        for c in self.control_points:
            c.set_data(
                pos=np.array([[0, 0]], dtype=np.float32),
                symbol="s",
                edge_color="red",
                size=6,
            )
            c.interactive = True
        self.freeze()

    def update_bounds(self) -> None:
        self._left = self.parent.bounds(0)[0]
        self._right = self.parent.bounds(0)[1]
        self._bottom = self.parent.bounds(1)[0]
        self._top = self.parent.bounds(1)[1]
        self.update_points()

    def update_points(self) -> None:
        self.control_points[0].set_data(pos=np.array([[self._left, self._top]]))
        self.control_points[1].set_data(pos=np.array([[self._right, self._top]]))
        self.control_points[2].set_data(pos=np.array([[self._right, self._bottom]]))
        self.control_points[3].set_data(pos=np.array([[self._left, self._bottom]]))

    def select(self, val: bool, obj: scene.visuals.Visual | None = None) -> None:
        self.visible(val)
        self.selected_idx = None
        self.opposed_idx = None

        if obj is not None:
            n_cp = len(self.control_points)
            for i in range(0, n_cp):
                c = self.control_points[i]
                if c == obj:
                    self.selected_idx = i
                    self.opposed_idx = int(i + n_cp / 2) % n_cp

    def start_move(self, pos: list[float]) -> None:
        pass

    def move(self, pos: list[float]) -> None:
        if self.selected_idx in [0, 3]:
            self._left = pos[0]
        else:
            self._right = pos[0]
        if self.selected_idx in [0, 1]:
            self._top = pos[1]
        else:
            self._bottom = pos[1]

        self.update_points()
        self.parent.update_from_controlpoints()

    @property
    def _width(self) -> float:
        return abs(self._left - self._right)

    @property
    def _height(self) -> float:
        return abs(self._top - self._bottom)

    def visible(self, v: bool) -> None:
        for c in self.control_points:
            c.visible = v

    def get_center(self) -> list[float]:
        return [0.5 * (self._left + self._right), 0.5 * (self._bottom + self._top)]

    def set_center(self, val: list[float]) -> None:
        # Translate rectangle
        center = self.get_center()
        dx = val[0] - center[0]
        self._left += dx
        self._right += dx
        dy = val[1] - center[1]
        self._bottom += dy
        self._top += dy
        self.update_points()


class EditRectVisual(scene.visuals.Compound):
    def __init__(
        self,
        center: list[float] | None = None,
        width: float = 20,
        height: float = 20,
        parent: scene.visuals.Visual | None = None,
    ) -> None:
        if center is None:
            center = [0, 0]
        super().__init__([], parent=parent)
        self.unfreeze()

        # Define polygon
        self.polygon = scene.visuals.Rectangle(
            center=center, width=width, height=height, radius=0, parent=self
        )
        self.polygon.interactive = True
        self.add_subvisual(self.polygon)

        # Define control points
        self.control_points = ControlPoints(parent=self)
        self.control_points.update_bounds()
        self.control_points.visible(False)

        # drag_reference defines the
        self.drag_reference = [0.0, 0.0]
        self.freeze()

    def select(self, val: bool, obj: scene.visuals.Visual | None = None) -> None:
        self.control_points.visible(val)

    def start_move(self, offset: list[float]) -> None:
        center = self.center
        self.drag_reference = [
            offset[0] - center[0],
            offset[1] - center[1],
        ]

    def move(self, pos: list[float]) -> None:
        shift = [
            pos[0] - self.drag_reference[0],
            pos[1] - self.drag_reference[1],
        ]
        self.center = shift

    def update_from_controlpoints(self) -> None:
        self.polygon.center = self.control_points.get_center()
        self.polygon.width = abs(self.control_points._width)
        self.polygon.height = abs(self.control_points._height)

    @property
    def center(self) -> list[float]:
        return self.control_points.get_center()

    @center.setter
    def center(self, val: list[float]) -> None:
        self.control_points.set_center(val[0:2])
        self.polygon.center = val[0:2]


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
    def __init__(self, roi: EditRectVisual) -> None:
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
        return self._roi.polygon.color

    @color.setter
    def color(self, color: cmap.Color | None = None) -> None:
        if color is None:
            color = cmap.Color("transparent")
        # NB: To enable dragging the shape within the border,
        # we require a positive alpha.
        alpha = max(color.alpha, 1e-6)
        self._roi.polygon.color = Color(color.hex, alpha=alpha)

    @property
    def border_color(self) -> Any:
        return self._roi.polygon.border_color

    @border_color.setter
    def border_color(self, color: cmap.Color | None = None) -> None:
        if color is None:
            color = cmap.Color("yellow")
        self._roi.polygon.border_color = Color(color.hex, alpha=color.alpha)

    def remove(self) -> None:
        self._roi.parent = None


class VispyViewerCanvas:
    """Vispy-based viewer for data.

    All vispy-specific code is encapsulated in this class (and non-vispy canvases
    could be swapped in if needed as long as they implement the same interface).
    """

    def __init__(self, set_info: Callable[[str], None]) -> None:
        self._set_info = set_info
        self._mode = CanvasMode.PAN_ZOOM
        self._canvas = scene.SceneCanvas()
        self._canvas.events.mouse_press.connect(self._on_mouse_press)
        self._canvas.events.mouse_move.connect(qthrottled(self._on_mouse_move, 60))
        self._current_shape: tuple[int, ...] = ()
        self._last_state: dict[Literal[2, 3], Any] = {}

        central_wdg: scene.Widget = self._canvas.central_widget
        self._view: scene.ViewBox = central_wdg.add_view()
        self._ndim: Literal[2, 3] | None = None

        self._selected_roi: scene.visuals.Visual = None

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

    def add_polygon(
        self,
        vertices: list[tuple[float, float]] | None = None,
        color: cmap.Color | None = None,
        border_color: cmap.Color | None = None,
    ) -> VispyRoiHandle:
        """Add a new Rectangular ROI node to the scene."""
        if color is None:
            color = cmap.Color("transparent")
        if border_color is None:
            border_color = cmap.Color("yellow")
        roi = EditRectVisual(
            center=[0, 0],
            width=1e-6,
            height=1e-6,
            parent=self._view.scene,
        )
        self._selected_roi = roi.control_points
        self._selected_roi.select(False, self._selected_roi.control_points[1])

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

    def _on_mouse_move(self, event: SceneMouseEvent) -> None:
        """Mouse moved on the canvas, display the pixel value and position."""
        self._update_cursor(event.pos)

        images = []
        # Get the images the mouse is over
        # FIXME: this is narsty ... there must be a better way to do this
        seen = set()
        try:
            while visual := self._canvas.visual_at(event.pos):
                if isinstance(visual, scene.visuals.Image):
                    images.append(visual)
                visual.interactive = False
                seen.add(visual)
        except Exception:
            return
        for visual in seen:
            visual.interactive = True
        if not images:
            return

        tform = images[0].get_transform("canvas", "visual")
        px, py, *_ = (int(x) for x in tform.map(event.pos))
        text = f"[{py}, {px}]"
        for c, img in enumerate(reversed(images)):
            with suppress(IndexError):
                value = img._data[py, px]
                if isinstance(value, (np.floating, float)):
                    value = f"{value:.2f}"
                text += f" {c}: {value}"
        self._set_info(text)

    def set_mode(self, mode: CanvasMode) -> None:
        self._mode = mode
        if self._mode is CanvasMode.EDIT_ROI:
            self._camera.interactive = False
        elif self._mode is CanvasMode.PAN_ZOOM:
            self._camera.interactive = True

    def _on_mouse_press(self, event: SceneMouseEvent) -> None:
        if self._mode == CanvasMode.EDIT_ROI:
            self._selected_roi.parent.center = self._canvas_point(event.pos)
            self._selected_roi.visible(True)
            handler = qthrottled(self._on_mouse_move_selection(self._selected_roi), 60)
            self._canvas.events.mouse_move.connect(handler)

            def disconnect(event: SceneMouseEvent) -> None:
                self._canvas.events.mouse_move.disconnect(handler)
                self._canvas.events.mouse_release.disconnect(disconnect)

            self._canvas.events.mouse_release.connect(disconnect)
        else:
            selected = self._canvas.visual_at(event.pos)
            if self._selected_roi is not None:
                self._selected_roi.select(False)
            if isinstance(selected.parent, EditRectVisual) or isinstance(
                selected.parent.parent, EditRectVisual
            ):
                self._selected_roi = selected.parent
                self._selected_roi.select(True, obj=selected)
                self._selected_roi.start_move(self._canvas_point(event.pos))
                self._camera.interactive = False
                handler = qthrottled(
                    self._on_mouse_move_selection(self._selected_roi), 60
                )
                self._canvas.events.mouse_move.connect(handler)

                def disconnect(event: SceneMouseEvent) -> None:
                    self._canvas.events.mouse_move.disconnect(handler)
                    self._canvas.events.mouse_release.disconnect(disconnect)
                    self._camera.interactive = True

                self._canvas.events.mouse_release.connect(disconnect)

    def _find_roi(
        self, selection: scene.visuals.Visual | None
    ) -> EditRectVisual | None:
        if selection is None:
            return None
        if isinstance(selection, EditRectVisual):
            return selection
        return self._find_roi(selection.parent)

    def _canvas_point(self, pos: list[float]) -> list[float]:
        tr = self._canvas.scene.node_transform(self._view.scene)
        return cast("list[float]", tr.map(pos)[:2])

    def _update_cursor(self, pos: list[float]) -> None:
        if self._mode is CanvasMode.EDIT_ROI:
            self.qwidget().setCursor(Qt.CrossCursor)
            return
        selected = self._canvas.visual_at(pos)

        if (
            self._selected_roi is not None
            and selected is not None
            and selected.parent is self._selected_roi
        ):
            self.qwidget().setCursor(Qt.SizeAllCursor)
            return

        self.qwidget().setCursor(Qt.ArrowCursor)

    def _on_mouse_move_selection(
        self, selection: scene.visuals.Visual
    ) -> Callable[[SceneMouseEvent], None]:
        def fooooooo(event: SceneMouseEvent) -> None:
            selection.move(self._canvas_point(event.pos))

        return fooooooo
