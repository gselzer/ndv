from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cmap as _cmap
import numpy as np
import numpy.typing as npt
import scenex as snx
from scenex.adaptors import get_adaptor_registry
from scenex.app import CursorType as SnxCursorType
from scenex.app import app as snx_app
from scenex.app import events
from scenex.utils import projections

from ndv._types import (
    CursorType,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
from ndv.models._lut_model import ClimsManual
from ndv.views._app import filter_mouse_events
from ndv.views.bases import HistogramCanvas

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ndv.models._lut_model import ClimPolicy


_AXIS = 40  # pixels reserved for each axis strip


class ScenexHistogramCanvas(HistogramCanvas):
    """HistogramCanvas backed by scenex."""

    def __init__(self) -> None:
        self._clims: tuple[float, float] = (0, 65535)
        self._gamma: float = 1.0
        self._grabbed: snx.Node | None = None
        self._initialized = False

        # Raw histogram data
        self._values: np.ndarray | None = None
        self._bins: np.ndarray | None = None
        self._log_base: float | None = None

        # Create canvas early so it's available before set_data
        self._canvas = snx.Canvas()
        self._canvas.visible = True

        # Create views
        self.x_view = snx.View(scene=snx.Scene(), camera=snx.Camera())
        self.view = snx.View(
            scene=snx.Scene(name="main scene"),
            camera=snx.Camera(interactive=True),
        )
        self.y_view = snx.View(scene=snx.Scene(), camera=snx.Camera())

        # Layout (pixel-based)
        self.x_view.layout.y_start = f"-{_AXIS}px"
        self._canvas.views.append(self.x_view)

        self.y_view.layout.x_end = f"{_AXIS}px"
        self.y_view.layout.y_end = f"-{_AXIS}px"
        self._canvas.views.append(self.y_view)

        self.view.layout.x_start = f"{_AXIS}px"
        self.view.layout.y_end = f"-{_AXIS}px"
        self._canvas.views.append(self.view)

        # Scene nodes (lazily initialized)
        self.x_axis: snx.Line | None = None
        self._tick_objects: list[snx.Text] = []
        self.y_axis: snx.Line | None = None
        self.y_max: snx.Text | None = None
        self.mesh: snx.Mesh | None = None
        self.highlight_line: snx.Line | None = None
        self.left_clim: snx.Line | None = None
        self.gamma_curve: snx.Line | None = None
        self.right_clim: snx.Line | None = None
        self.gamma_handle: snx.Points | None = None
        self.controls: snx.Scene | None = None

        # Set up mouse event handling on the native widget
        native = get_adaptor_registry().get_adaptor(self._canvas)._snx_get_native()
        self._disconnect_mouse_events = filter_mouse_events(native, self)

        # Also use the scenex event filter to intercept view-level events
        # (for clim handle interaction which needs world-ray / world coords)
        self.view.set_event_filter(self._on_main_view)

    # -- Viewable methods -- #

    def frontend_widget(self) -> Any:
        return get_adaptor_registry().get_adaptor(self._canvas)._snx_get_native()

    def set_visible(self, visible: bool) -> None:
        self._canvas.visible = visible

    def close(self) -> None:
        self._disconnect_mouse_events()

    # -- GraphicsCanvas abstract methods -- #

    def refresh(self) -> None:
        try:
            self._canvas.request_draw()
        except AttributeError:
            pass

    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = 0,
    ) -> None:
        if not self._initialized:
            return
        projections.zoom_to_fit(self.view, "orthographic", zoom_factor=1)
        self.x_view.camera.projection = projections.orthographic(1, 1, 1)
        self.y_view.camera.projection = projections.orthographic(1, 1, 1)
        self.x_view.camera.transform = snx.Transform().translated((0.5, -0.5, 0))
        self.y_view.camera.transform = snx.Transform().translated((-0.5, 0.5, 0))

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        cam = self.view.camera
        w = self._canvas.width
        h = self._canvas.height
        if not w or not h:
            return (0.0, 0.0, 0.0)
        ndc_x = 2.0 * pos_xy[0] / w - 1.0
        ndc_y = 1.0 - 2.0 * pos_xy[1] / h
        world = cam.transform.map(cam.projection.imap((ndc_x, ndc_y)))
        return (float(world[0]), float(world[1]), 0.0)

    def elements_at(self, pos_xy: tuple[float, float]) -> list:
        return []

    # -- LUTView abstract methods -- #

    def set_channel_name(self, name: str) -> None:
        pass

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        if isinstance(policy, ClimsManual):
            self.set_clims((policy.min, policy.max))

    def set_colormap(self, colormap: _cmap.Colormap) -> None:
        if self.mesh is not None:
            self.mesh.color = snx.UniformColor(
                color=colormap.color_stops[-1].color
            )

    def set_clims(self, clims: tuple[float, float]) -> None:
        self._clims = clims
        if self.controls is not None:
            self.controls.transform = (
                snx.Transform()
                .scaled((self._clims[1] - self._clims[0], 1, 1))
                .translated((self._clims[0], 0, 0))
            )

    def set_clim_bounds(
        self, bounds: tuple[float | None, float | None] = (None, None)
    ) -> None:
        pass

    def set_channel_visible(self, visible: bool) -> None:
        pass

    # -- HistogramCanvas methods -- #

    def set_gamma(self, gamma: float) -> None:
        self._gamma = gamma
        self._update_lut_line()

    def set_data(self, values: np.ndarray, bin_edges: np.ndarray) -> None:
        """Set histogram data (pre-computed counts and bin edges)."""
        self._initialize_views()

        self._values = np.copy(values)
        self._bins = np.copy(bin_edges)

        if self._log_base is not None:
            display_values = np.log(values + 1) / np.log(self._log_base)
        else:
            display_values = values

        if mesh := self.mesh:
            mesh.vertices, mesh.faces = _hist_counts_to_mesh(
                display_values, bin_edges, False
            )

        self._update_y_axis()

    def set_log_base(self, base: float | None) -> None:
        if self._values is None or self.mesh is None:
            self._log_base = base
            return

        self._log_base = base
        values = self._values
        if base is not None:
            display_values = np.log(values + 1) / np.log(base)
        else:
            display_values = values

        if self._bins is not None:
            self.mesh.vertices, self.mesh.faces = _hist_counts_to_mesh(
                display_values, self._bins, False
            )
        self._update_y_axis()

    def set_vertical(self, vertical: bool) -> None:
        # Not supported in this scenex implementation; silently ignore
        pass

    def highlight(self, value: float | None) -> None:
        if self.highlight_line is None:
            return
        self.highlight_line.visible = value is not None
        if value is not None:
            self.highlight_line.transform = snx.Transform().translated((value, 0, 0))

    # -- Mouseable overrides (for ROI cursor; minimal impl) -- #

    def get_cursor(self, event: MouseMoveEvent) -> CursorType | None:
        return None

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        return False

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        return False

    # -- Scenex event filter (handles clim/gamma dragging) -- #

    def _on_main_view(self, event: events.Event) -> bool:
        if not self._initialized or self.controls is None:
            return False

        if isinstance(event, events.MousePressEvent):
            intersections = [
                node
                for node, _dist in event.world_ray.intersections(self.controls)
                if node.interactive
            ]
            if intersections:
                self._grabbed = intersections[0]
                self.view.camera.interactive = False
        elif isinstance(event, events.MouseDoublePressEvent):
            intersections = [
                node
                for node, _dist in event.world_ray.intersections(self.controls)
                if node.interactive
            ]
            if self.gamma_handle in intersections and (model := self.model):
                model.gamma = 1

        if isinstance(event, events.MouseMoveEvent):
            if self._grabbed is self.left_clim:
                new_left = min(event.world_ray.origin[0], self._clims[1])
                if self._bins is not None:
                    new_left = max(new_left, self._bins[0])
                if model := self.model:
                    model.clims = ClimsManual(min=new_left, max=self._clims[1])
            elif self._grabbed is self.right_clim:
                new_right = max(self._clims[0], event.world_ray.origin[0])
                if self._bins is not None:
                    new_right = min(new_right, self._bins[-1])
                if model := self.model:
                    model.clims = ClimsManual(min=self._clims[0], max=new_right)
            elif self._grabbed is self.gamma_handle:
                if model := self.model:
                    model.gamma = -np.log2(
                        max(event.world_ray.origin[1], 1e-10)
                    )
            elif self._grabbed is None:
                intersections = [
                    node
                    for node, _dist in event.world_ray.intersections(self.controls)
                    if node.interactive
                ]
                try:
                    if self.right_clim in intersections or self.left_clim in intersections:
                        snx_app().set_cursor(self._canvas, SnxCursorType.H_ARROW)
                    elif self.gamma_handle in intersections:
                        snx_app().set_cursor(self._canvas, SnxCursorType.V_ARROW)
                    else:
                        snx_app().set_cursor(self._canvas, SnxCursorType.DEFAULT)
                except Exception:
                    pass

        if isinstance(event, (events.MouseReleaseEvent, events.MouseLeaveEvent)):
            self._grabbed = None
            self.view.camera.interactive = True

        return False

    # -- Private helpers -- #

    def _initialize_views(self) -> None:
        if self._initialized:
            return

        # X axis view
        self.x_axis = snx.Line(
            vertices=np.array([[0, 0, 0], [1, 0, 0]]),
            width=2,
            color=snx.UniformColor(color=_cmap.Color("white")),
        )
        self.x_view.scene.add_child(self.x_axis)

        # Pre-create tick objects
        for _ in range(10):
            tick_line = snx.Line(
                vertices=np.array([[0, 0, 0], [0, -0.1, 0]]),
                width=1,
                color=snx.UniformColor(color=_cmap.Color("white")),
                transform=snx.Transform().translated((0, 0.4, 0)),
            )
            tick_text = snx.Text(text="0", children=[tick_line], antialias=True)
            self._tick_objects.append(tick_text)

        # Y axis view
        self.y_axis = snx.Line(
            vertices=np.array([[0, 0, 0], [0, 1, 0]]),
            width=2,
            color=snx.UniformColor(color=_cmap.Color("white")),
        )
        self.y_max = snx.Text(
            text="1",
            transform=snx.Transform().translated((-0.5, 0.95)),
            antialias=True,
        )
        self.y_view.scene.add_child(self.y_axis)
        self.y_view.scene.add_child(self.y_max)

        # Main histogram mesh
        self.mesh = snx.Mesh(
            vertices=np.zeros((1, 3), dtype=np.float32),
            faces=np.zeros((1, 3), dtype=np.uint16),
            color=snx.UniformColor(color=_cmap.Color("steelblue")),
            order=0,
        )

        self.highlight_line = snx.Line(
            vertices=np.array([[0, 0, 0], [0, 1, 0]]),
            width=2,
            color=snx.UniformColor(color=_cmap.Color("yellow")),
            visible=False,
        )

        # Clim and gamma controls
        self.left_clim = snx.Line(name="left clim", interactive=True, order=1)
        self.gamma_curve = snx.Line(name="gamma curve", interactive=False, order=1)
        self.right_clim = snx.Line(name="right clim", interactive=True, order=1)
        self.gamma_handle = snx.Points(
            name="gamma handle",
            vertices=np.array([[0.5, 0.5, 0]]),
            size=8,
            scaling="fixed",
            face_color=snx.UniformColor(color=_cmap.Color("white")),
            edge_color=snx.UniformColor(color=_cmap.Color("black")),
            interactive=True,
            order=2,
        )

        self._create_static_clim_lines()
        self._update_lut_line()

        self.controls = snx.Scene(
            name="controls scene",
            children=[
                self.left_clim,
                self.gamma_curve,
                self.right_clim,
                self.gamma_handle,
            ],
            interactive=True,
        )

        self.view.scene.add_child(self.mesh)
        self.controls.order = 1
        self.view.scene.add_child(self.controls)
        self.highlight_line.order = 2
        self.view.scene.add_child(self.highlight_line)

        self.view.camera.controller = snx.PanZoom(lock_y=True)
        self.view.camera.events.transform.connect(self._update_x_axis)
        self.view.camera.events.projection.connect(self._update_x_axis)
        self._canvas.events.width.connect(self._update_x_axis)

        self._initialized = True
        self.set_clims(self._clims)

    def _create_static_clim_lines(self) -> None:
        dark = _cmap.Color((0.4, 0.4, 0.4))
        light = _cmap.Color((0.7, 0.7, 0.7))

        left_verts = np.column_stack(([0, 0, 0], [1, 0.5, 0], np.zeros(3))).astype(
            np.float32
        )
        if line := self.left_clim:
            line.vertices = left_verts
            line.color = snx.VertexColors(color=[dark, light, dark])

        right_verts = np.column_stack(([1, 1, 1], [1, 0.5, 0], np.zeros(3))).astype(
            np.float32
        )
        if line := self.right_clim:
            line.vertices = right_verts
            line.color = snx.VertexColors(color=[dark, light, dark])

    def _update_lut_line(self) -> None:
        if self.gamma_curve is None or self.gamma_handle is None:
            return
        npoints = 256
        gamma = self._gamma
        gamma_x = np.linspace(0, 1, npoints)
        gamma_y = np.linspace(0, 1, npoints) ** gamma
        gamma_z = np.zeros(npoints)
        self.gamma_curve.vertices = np.column_stack(
            (gamma_x, gamma_y, gamma_z)
        ).astype(np.float32)
        gamma_colors = [
            _cmap.Color(c)
            for c in np.linspace(0.2, 0.8, npoints).repeat(3).reshape(-1, 3)
        ]
        self.gamma_curve.color = snx.VertexColors(color=gamma_colors)
        self.gamma_handle.transform = snx.Transform().translated((0, 0.5**gamma - 0.5))

    def _update_y_axis(self) -> None:
        if self.mesh is None or self.y_max is None:
            return
        bb = self.mesh.bounding_box
        if bb is None:
            return
        max_val = bb[1][1]
        self.mesh.transform = snx.Transform().scaled((1, 0.95 / max(max_val, 1), 1))
        self.y_max.text = f"{max_val:.2f}"

    def _update_x_axis(self) -> None:
        if not self._initialized:
            return
        cam = self.view.camera
        left, *_ = cam.transform.map(cam.projection.imap((-1, 0)))
        right, *_ = cam.transform.map(cam.projection.imap((1, 0)))
        self._clear_ticks()
        tick_step = _calculate_tick_step(left, right)
        positions = _get_tick_positions(left, right, tick_step)
        try:
            _x, _y, w, _h = self._canvas.rect_for(self.x_view)
            start = _AXIS / w if w else 0.1
        except Exception:
            start = 0.1
        for i, tick_val in enumerate(positions):
            if i >= len(self._tick_objects):
                break
            norm_pos = (
                start + (tick_val - left) / (right - left) * (1 - start)
                if right != left
                else 0.5
            )
            tick_obj = self._tick_objects[i]
            tick_obj.text = f"{tick_val:.0f}"
            tick_obj.transform = snx.Transform().translated((norm_pos, -0.5, 0))
            self.x_view.scene.add_child(tick_obj)

    def _clear_ticks(self) -> None:
        for tick_obj in self._tick_objects:
            if tick_obj in self.x_view.scene.children:
                self.x_view.scene.remove_child(tick_obj)


# -- Module-level helpers -- #


def _hist_counts_to_mesh(
    values: Sequence[float] | npt.NDArray,
    bin_edges: Sequence[float] | npt.NDArray,
    vertical: bool = False,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
    """Convert histogram counts to mesh vertices and faces."""
    n_edges = len(bin_edges)
    X, Y = (1, 0) if vertical else (0, 1)
    vertices = np.zeros((3 * n_edges - 2, 3), np.float32)
    vertices[:, X] = np.repeat(bin_edges, 3)[1:-1]
    vertices[1::3, Y] = values
    vertices[2::3, Y] = values
    vertices[vertices == float("-inf")] = 0
    faces = np.zeros((2 * n_edges - 2, 3), np.uint32)
    offsets = 3 * np.arange(n_edges - 1, dtype=np.uint32)[:, np.newaxis]
    faces[::2] = np.array([0, 2, 1]) + offsets
    faces[1::2] = np.array([2, 0, 3]) + offsets
    return vertices, faces


def _calculate_tick_step(min_val: float, max_val: float, target_ticks: int = 5) -> float:
    from math import floor, log10

    if max_val <= min_val:
        return 1.0
    range_val = max_val - min_val
    approx_step = range_val / target_ticks
    power10 = 10 ** floor(log10(approx_step))
    for multiplier in [1, 2, 2.5, 5, 10]:
        step = multiplier * power10
        if step >= approx_step:
            return step
    return power10


def _get_tick_positions(
    min_val: float, max_val: float, step: float
) -> list[float]:
    from math import ceil, floor

    if step <= 0:
        return [min_val, max_val]
    first_tick = ceil(min_val / step) * step
    last_tick = floor(max_val / step) * step
    intermediate: list[float] = []
    current = first_tick
    while current <= last_tick and len(intermediate) < 20:
        intermediate.append(current)
        current += step
    min_distance = step * 0.15
    filtered = [
        v
        for v in intermediate
        if abs(v - min_val) >= min_distance and abs(v - max_val) >= min_distance
    ]
    all_ticks = [min_val, *filtered, max_val]
    seen: set[float] = set()
    unique: list[float] = []
    for tick in all_ticks:
        if tick not in seen:
            seen.add(tick)
            unique.append(tick)
    return unique
