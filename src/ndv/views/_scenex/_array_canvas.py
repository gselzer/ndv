from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from weakref import ReferenceType

import cmap as _cmap
import numpy as np
import scenex as snx
from scenex.adaptors import get_adaptor_registry
from scenex.model import BlendMode
from scenex.utils import projections

from ndv._types import (
    CursorType,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
from ndv.models._lut_model import ClimsManual
from ndv.models._viewer_model import InteractionMode
from ndv.views._app import filter_mouse_events
from ndv.views.bases import ArrayCanvas
from ndv.views.bases._graphics._canvas_elements import (
    CanvasElement,
    ImageHandle,
    RectangularROIHandle,
    ROIMoveMode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ndv.models._lut_model import ClimPolicy
    from ndv.models._viewer_model import ArrayViewerModel


class ScenexImageHandle(ImageHandle):
    """ImageHandle wrapping a scenex Image or Volume visual."""

    def __init__(self, visual: snx.Image | snx.Volume) -> None:
        self._visual = visual
        self._cmap: _cmap.Colormap = _cmap.Colormap("grays")

    # -- ImageHandle abstract methods -- #

    def data(self) -> np.ndarray:
        return self._visual.data  # type: ignore[no-any-return]

    def set_data(self, data: np.ndarray) -> None:
        self._visual.data = data

    def clims(self) -> tuple[float, float]:
        return self._visual.clims  # type: ignore[no-any-return]

    def set_clims(self, clims: tuple[float, float]) -> None:
        self._visual.clims = clims

    def gamma(self) -> float:
        return self._visual.gamma  # type: ignore[no-any-return]

    def set_gamma(self, gamma: float) -> None:
        # Clamp to scenex-supported range
        self._visual.gamma = max(1e-6, min(gamma, 2))

    def colormap(self) -> _cmap.Colormap:
        return self._cmap

    def set_colormap(self, cmap: _cmap.Colormap) -> None:
        self._cmap = cmap
        self._visual.cmap = cmap

    # -- CanvasElement abstract methods -- #

    def visible(self) -> bool:
        return bool(self._visual.visible)

    def set_visible(self, visible: bool) -> None:
        self._visual.visible = visible

    def can_select(self) -> bool:
        return False

    def selected(self) -> bool:
        return False

    def set_selected(self, selected: bool) -> None:
        pass

    def remove(self) -> None:
        self._visual.parent = None

    # -- Mouseable methods (defaults are fine; images aren't interactable) -- #

    def get_cursor(self, event: MouseMoveEvent) -> CursorType | None:
        return None

    # -- LUTView methods (non-abstract overrides) -- #

    def set_channel_name(self, name: str) -> None:
        self._visual.name = name

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        if isinstance(policy, ClimsManual):
            self.set_clims((policy.min, policy.max))

    def set_channel_visible(self, visible: bool) -> None:
        self.set_visible(visible)

    def frontend_widget(self) -> Any:
        return None

    def close(self) -> None:
        self.remove()


class ScenexRectangle(RectangularROIHandle):
    """RectangularROIHandle backed by scenex Line and Points visuals."""

    def __init__(
        self,
        scene: snx.Scene,
        canvas_to_world: Any,  # callable: (x, y) -> (wx, wy, wz)
        pixel_to_world_scale: Any,  # callable: (n_pixels) -> float
    ) -> None:
        self._canvas_to_world = canvas_to_world
        self._pixel_to_world_scale = pixel_to_world_scale
        self._selected = False
        self._move_mode: ROIMoveMode | None = None
        # anchor has different meanings depending on _move_mode
        self._move_anchor: tuple[float, float] = (0.0, 0.0)

        # Border: 5 vertices to form a closed rectangle (last == first)
        self._border_verts = np.zeros((5, 3), dtype=np.float32)
        self._border = snx.Line(
            vertices=self._border_verts,
            interactive=True,
        )
        scene.add_child(self._border)

        # Corner handles (4 corners)
        self._handle_positions = np.zeros((4, 2), dtype=np.float64)
        self._handle_pts = snx.Points(
            coords=np.zeros((4, 3), dtype=np.float32),
            size=8,
            scaling=False,
            interactive=True,
        )
        scene.add_child(self._handle_pts)

        # Default colors
        self._border.color = snx.UniformColor(color=_cmap.Color("yellow"))

        self._scene_ref = scene
        self._min: tuple[float, float] = (0.0, 0.0)
        self._max: tuple[float, float] = (1.0, 1.0)

        self.set_visible(False)

    def can_select(self) -> bool:
        return True

    def selected(self) -> bool:
        return self._selected

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._handle_pts.visible = selected and self.visible()

    def set_fill(self, color: _cmap.Color) -> None:
        # scenex Line doesn't have a fill; skip
        pass

    def set_border(self, color: _cmap.Color) -> None:
        self._border.color = snx.UniformColor(color=color)

    def set_handles(self, color: _cmap.Color) -> None:
        self._handle_pts.face_color = snx.UniformColor(color=color)

    def set_bounding_box(
        self, minimum: tuple[float, float], maximum: tuple[float, float]
    ) -> None:
        x1 = float(min(minimum[0], maximum[0]))
        y1 = float(min(minimum[1], maximum[1]))
        x2 = float(max(minimum[0], maximum[0]))
        y2 = float(max(minimum[1], maximum[1]))
        self._min = (x1, y1)
        self._max = (x2, y2)

        # Update border (closed rectangle)
        self._border_verts[:] = [
            [x1, y1, 0],
            [x2, y1, 0],
            [x2, y2, 0],
            [x1, y2, 0],
            [x1, y1, 0],
        ]
        self._border.vertices = self._border_verts

        # Update handle positions (4 corners: TL, TR, BR, BL)
        self._handle_positions[:] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        handle_coords = np.column_stack(
            [self._handle_positions, np.zeros(4, dtype=np.float32)]
        )
        self._handle_pts.coords = handle_coords.astype(np.float32)

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        self.set_selected(True)
        world_pos = self._canvas_to_world((event.x, event.y))[:2]
        drag_idx = self._handle_under(world_pos)
        if drag_idx is not None:
            opposite_idx = (drag_idx + 2) % 4
            self._move_mode = ROIMoveMode.HANDLE
            self._move_anchor = tuple(self._handle_positions[opposite_idx].copy())  # type: ignore[assignment]
        else:
            self._move_mode = ROIMoveMode.TRANSLATE
            self._move_anchor = world_pos
        return False

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        world_pos = self._canvas_to_world((event.x, event.y))[:2]
        if self._move_mode == ROIMoveMode.HANDLE:
            self.boundingBoxChanged.emit((world_pos, self._move_anchor))
        elif self._move_mode == ROIMoveMode.TRANSLATE:
            dx = world_pos[0] - self._move_anchor[0]
            dy = world_pos[1] - self._move_anchor[1]
            new_min = (self._min[0] + dx, self._min[1] + dy)
            new_max = (self._max[0] + dx, self._max[1] + dy)
            self.boundingBoxChanged.emit((new_min, new_max))
            self._move_anchor = world_pos
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        self._move_mode = None
        return False

    def get_cursor(self, event: MouseMoveEvent) -> CursorType | None:
        world_pos = self._canvas_to_world((event.x, event.y))[:2]
        if self._handle_under(world_pos) is not None:
            center = (
                (self._min[0] + self._max[0]) / 2,
                (self._min[1] + self._max[1]) / 2,
            )
            if world_pos[0] < center[0] and world_pos[1] < center[1]:
                return CursorType.FDIAG_ARROW
            if world_pos[0] > center[0] and world_pos[1] > center[1]:
                return CursorType.FDIAG_ARROW
            return CursorType.BDIAG_ARROW
        return CursorType.ALL_ARROW

    def visible(self) -> bool:
        return bool(self._border.visible)

    def set_visible(self, visible: bool) -> None:
        self._border.visible = visible
        self._handle_pts.visible = visible and self.selected()

    def remove(self) -> None:
        self._border.parent = None
        self._handle_pts.parent = None

    def _handle_under(self, world_pos: Sequence[float]) -> int | None:
        """Return index of handle at world_pos, or None."""
        tol = self._pixel_to_world_scale(5)
        for i, p in enumerate(self._handle_positions):
            if abs(p[0] - world_pos[0]) <= tol and abs(p[1] - world_pos[1]) <= tol:
                return i
        return None

    def _hit_test(self, world_pos: Sequence[float], tol: float) -> bool:
        """Return True if world_pos is near the ROI border or handles."""
        x1, y1 = self._min
        x2, y2 = self._max
        cx, cy = world_pos[0], world_pos[1]
        # Check handles
        for p in self._handle_positions:
            if abs(p[0] - cx) <= tol and abs(p[1] - cy) <= tol:
                return True
        # Check border edges
        if x1 - tol <= cx <= x2 + tol and y1 - tol <= cy <= y2 + tol:
            near_x = abs(cx - x1) <= tol or abs(cx - x2) <= tol
            near_y = abs(cy - y1) <= tol or abs(cy - y2) <= tol
            if near_x or near_y:
                return True
        return False


class ScenexArrayCanvas(ArrayCanvas):
    """ArrayCanvas implementation backed by scenex."""

    def __init__(self, viewer_model: ArrayViewerModel) -> None:
        self._viewer = viewer_model

        self.view = snx.View(
            scene=snx.Scene(interactive=True),
            camera=snx.Camera(interactive=True),
            on_resize=snx.Letterbox(),
        )
        self._canvas = snx.Canvas(
            width=600, height=600, views=[self.view], visible=True
        )

        self._ndim: Literal[2, 3] | None = None

        # All active ROI handles (for hit testing in elements_at)
        self._roi_handles: list[ScenexRectangle] = []
        self._selection: CanvasElement | None = None
        self._last_roi_created: ReferenceType[ScenexRectangle] | None = None

        # Set up mouse event handling via filter on the native widget
        native = get_adaptor_registry().get_adaptor(self._canvas)._snx_get_native()
        self._disconnect_mouse_events = filter_mouse_events(native, self)

    # -- Viewable methods -- #

    def frontend_widget(self) -> Any:
        return get_adaptor_registry().get_adaptor(self._canvas)._snx_get_native()

    def set_visible(self, visible: bool) -> None:
        self._canvas.visible = visible

    def close(self) -> None:
        self._disconnect_mouse_events()

    # -- ArrayCanvas abstract methods -- #

    def set_ndim(self, ndim: Literal[2, 3]) -> None:
        if ndim == self._ndim:
            return
        self._ndim = ndim
        if ndim == 2:
            self.view.camera.controller = snx.PanZoom()
        else:
            self.view.camera.controller = snx.Orbit()
        self._reset_zoom()

    def add_image(self, data: np.ndarray | None = None) -> ScenexImageHandle:
        img = snx.Image(
            data=data if data is not None else np.zeros((1, 1), dtype=np.float32),
            blending=BlendMode.ADDITIVE,
            interactive=True,
        )
        self.view.scene.add_child(img)
        handle = ScenexImageHandle(img)
        if data is not None:
            self.set_range()
        return handle

    def add_volume(self, data: np.ndarray | None = None) -> ScenexImageHandle:
        vol = snx.Volume(
            data=data if data is not None else np.zeros((1, 1, 1), dtype=np.float32),
            blending=BlendMode.ADDITIVE,
            interactive=True,
        )
        self.view.scene.add_child(vol)
        handle = ScenexImageHandle(vol)
        if data is not None:
            self.set_range()
        return handle

    def add_bounding_box(self) -> ScenexRectangle:
        roi = ScenexRectangle(
            self.view.scene,
            self.canvas_to_world,
            self._pixel_to_world_scale,
        )
        roi.set_visible(False)
        self._roi_handles.append(roi)
        self._last_roi_created = ReferenceType(roi)
        return roi

    def set_scales(self, scales: tuple[float, ...]) -> None:
        if not scales:
            return
        vis_scales = list(reversed(scales))
        while len(vis_scales) < 3:
            vis_scales.append(1.0)
        sx, sy, sz = vis_scales[0], vis_scales[1], vis_scales[2]
        for child in list(self.view.scene.children):
            if isinstance(child, (snx.Image, snx.Volume)):
                child.transform = snx.Transform().scaled((sx, sy, sz))
        self.set_range()

    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = 0.01,
    ) -> None:
        self._reset_zoom(zoom_factor=1 - margin)

    def refresh(self) -> None:
        # scenex auto-renders on data changes; this is a best-effort hint
        try:
            self._canvas.request_draw()
        except AttributeError:
            pass

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        cam = self.view.camera
        w = self._canvas.width
        h = self._canvas.height
        if not w or not h:
            return (0.0, 0.0, 0.0)
        ndc_x = 2.0 * pos_xy[0] / w - 1.0
        ndc_y = 1.0 - 2.0 * pos_xy[1] / h  # flip y (canvas y=0 is top)
        world = cam.transform.map(cam.projection.imap((ndc_x, ndc_y)))
        return (float(world[0]), float(world[1]), 0.0)

    def elements_at(self, pos_xy: tuple[float, float]) -> list[CanvasElement]:
        """Find canvas elements at the given canvas position."""
        world_pos = self.canvas_to_world(pos_xy)[:2]
        tol = self._pixel_to_world_scale(5)
        elements: list[CanvasElement] = []
        for roi in self._roi_handles:
            if roi.visible() and roi._hit_test(world_pos, tol):
                elements.append(roi)
        return elements

    # -- Mouseable overrides -- #

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        if self._selection:
            self._selection.set_selected(False)
            self._selection = None

        canvas_pos = (event.x, event.y)
        world_pos = self.canvas_to_world(canvas_pos)[:2]

        if self._viewer.interaction_mode == InteractionMode.CREATE_ROI:
            if self._last_roi_created is None:
                raise ValueError("No ROI to create!")
            if new_roi := self._last_roi_created():
                self._last_roi_created = None
                _min = world_pos
                _max = (world_pos[0] + 1, world_pos[1] + 1)
                new_roi.boundingBoxChanged.emit((_min, _max))
                new_roi.set_visible(True)
                new_roi.set_selected(True)
            self._viewer.interaction_mode = InteractionMode.PAN_ZOOM

        for element in self.elements_at(canvas_pos):
            if element.can_select():
                self._selection = element
                self._selection.on_mouse_press(event)
                return False

        return False

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        if event.btn == MouseButton.LEFT:
            if self._selection and self._selection.selected():
                self._selection.on_mouse_move(event)
                return True
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        if self._selection:
            self._selection.on_mouse_release(event)
        return False

    def get_cursor(self, event: MouseMoveEvent) -> CursorType:
        if self._viewer.interaction_mode == InteractionMode.CREATE_ROI:
            return CursorType.CROSS
        for element in self.elements_at((event.x, event.y)):
            if cursor := element.get_cursor(event):
                return cursor
        return CursorType.DEFAULT

    # -- Private helpers -- #

    def _reset_zoom(self, zoom_factor: float = 0.9) -> None:
        controller = self.view.camera.controller
        if self._ndim == 3:
            projections.zoom_to_fit(
                self.view,
                type="perspective",
                zoom_factor=zoom_factor,
                preserve_aspect_ratio=True,
            )
            if isinstance(controller, snx.Orbit):
                if bb := self.view.scene.bounding_box:
                    controller.center = np.mean(bb, axis=0)
                else:
                    controller.center = (0, 0, 0)
        else:
            projections.zoom_to_fit(
                self.view,
                type="orthographic",
                zoom_factor=zoom_factor,
                preserve_aspect_ratio=True,
            )

    def _pixel_to_world_scale(self, n_pixels: float) -> float:
        """Approximate world-space size of `n_pixels` canvas pixels."""
        w = self._canvas.width
        if not w:
            return 1.0
        p0 = self.canvas_to_world((0.0, 0.0))
        p1 = self.canvas_to_world((n_pixels, 0.0))
        return max(abs(p1[0] - p0[0]), 1e-10)
