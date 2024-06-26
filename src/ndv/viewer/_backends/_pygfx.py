from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np
import pygfx
import pylinalg as la
from qtpy.QtCore import QSize
from wgpu.gui.qt import QWgpuCanvas

if TYPE_CHECKING:
    import cmap
    from pygfx.materials import ImageBasicMaterial
    from pygfx.resources import Texture
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QWidget


class PyGFXImageHandle:
    def __init__(self, image: pygfx.Image | pygfx.Volume, render: Callable) -> None:
        self._image = image
        self._render = render
        self._grid = cast("Texture", image.geometry.grid)
        self._material = cast("ImageBasicMaterial", image.material)

    @property
    def data(self) -> np.ndarray:
        return self._grid.data  # type: ignore [no-any-return]

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self._grid.data[:] = data
        self._grid.update_range((0, 0, 0), self._grid.size)

    @property
    def visible(self) -> bool:
        return bool(self._image.visible)

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._image.visible = visible
        self._render()

    @property
    def clim(self) -> Any:
        return self._material.clim

    @clim.setter
    def clim(self, clims: tuple[float, float]) -> None:
        self._material.clim = clims
        self._render()

    @property
    def cmap(self) -> cmap.Colormap:
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: cmap.Colormap) -> None:
        self._cmap = cmap
        self._material.map = cmap.to_pygfx()
        self._render()

    def remove(self) -> None:
        if (par := self._image.parent) is not None:
            par.remove(self._image)


class _QWgpuCanvas(QWgpuCanvas):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sup_mouse_event = self._subwidget._mouse_event
        self._subwidget._mouse_event = self._mouse_event

    def sizeHint(self) -> QSize:
        return QSize(512, 512)

    def _mouse_event(
        self, event_type: str, event: QEvent, *args: Any, **kwargs: Any
    ) -> None:
        self._sup_mouse_event(event_type, event, *args, **kwargs)
        event.ignore()


class PyGFXViewerCanvas:
    """pygfx-based canvas wrapper."""

    def __init__(self) -> None:
        self._current_shape: tuple[int, ...] = ()
        self._last_state: dict[Literal[2, 3], Any] = {}

        self._canvas = _QWgpuCanvas(size=(512, 512))
        self._renderer = pygfx.renderers.WgpuRenderer(self._canvas)
        try:
            # requires https://github.com/pygfx/pygfx/pull/752
            self._renderer.blend_mode = "additive"
        except ValueError:
            warnings.warn(
                "This version of pygfx does not yet support additive blending.",
                stacklevel=3,
            )
            self._renderer.blend_mode = "weighted_depth"

        self._scene = pygfx.Scene()
        self._camera: pygfx.Camera | None = None
        self._ndim: Literal[2, 3] | None = None

    def qwidget(self) -> QWidget:
        return cast("QWidget", self._canvas)

    def set_ndim(self, ndim: Literal[2, 3]) -> None:
        """Set the number of dimensions of the displayed data."""
        if ndim == self._ndim:
            return
        elif self._ndim is not None and self._camera is not None:
            # remember the current state before switching to the new camera
            self._last_state[self._ndim] = self._camera.get_state()

        self._ndim = ndim
        if ndim == 3:
            self._camera = cam = pygfx.PerspectiveCamera(0, 1)
            cam.show_object(self._scene, up=(0, -1, 0), view_dir=(0, 0, 1))
            controller = pygfx.OrbitController(cam, register_events=self._renderer)
            zoom = "zoom"
            # FIXME: there is still an issue with rotational centration.
            # the controller is not rotating around the middle of the volume...
            # but I think it might actually be a pygfx issue... the critical state
            # seems to be somewhere outside of the camera's get_state dict.
        else:
            self._camera = cam = pygfx.OrthographicCamera(512, 512)
            cam.local.scale_y = -1
            cam.local.position = (256, 256, 0)
            controller = pygfx.PanZoomController(cam, register_events=self._renderer)
            zoom = "zoom_to_point"

        self._controller = controller
        # increase zoom wheel gain
        self._controller.controls.update({"wheel": (zoom, "push", -0.005)})

        # restore the previous state if it exists
        if state := self._last_state.get(ndim):
            cam.set_state(state)

    def add_image(
        self, data: np.ndarray | None = None, cmap: cmap.Colormap | None = None
    ) -> PyGFXImageHandle:
        """Add a new Image node to the scene."""
        tex = pygfx.Texture(data, dim=2)
        image = pygfx.Image(
            pygfx.Geometry(grid=tex),
            # depth_test=False for additive-like blending
            pygfx.ImageBasicMaterial(depth_test=False),
        )
        self._scene.add(image)

        if data is not None:
            self._current_shape, prev_shape = data.shape, self._current_shape
            if not prev_shape:
                self.set_range()

        # FIXME: I suspect there are more performant ways to refresh the canvas
        # look into it.
        handle = PyGFXImageHandle(image, self.refresh)
        if cmap is not None:
            handle.cmap = cmap
        return handle

    def add_volume(
        self, data: np.ndarray | None = None, cmap: cmap.Colormap | None = None
    ) -> PyGFXImageHandle:
        tex = pygfx.Texture(data, dim=3)
        vol = pygfx.Volume(
            pygfx.Geometry(grid=tex),
            # depth_test=False for additive-like blending
            pygfx.VolumeRayMaterial(interpolation="nearest", depth_test=False),
        )
        self._scene.add(vol)

        if data is not None:
            vol.local_position = [-0.5 * i for i in data.shape[::-1]]
            self._current_shape, prev_shape = data.shape, self._current_shape
            if len(prev_shape) != 3:
                self.set_range()

        # FIXME: I suspect there are more performant ways to refresh the canvas
        # look into it.
        handle = PyGFXImageHandle(vol, self.refresh)
        if cmap is not None:
            handle.cmap = cmap
        return handle

    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = 0.05,
    ) -> None:
        """Update the range of the PanZoomCamera.

        When called with no arguments, the range is set to the full extent of the data.
        """
        if not self._scene.children or self._camera is None:
            return

        cam = self._camera
        cam.show_object(self._scene)

        width, height, depth = np.ptp(self._scene.get_world_bounding_box(), axis=0)
        if width < 0.01:
            width = 1
        if height < 0.01:
            height = 1
        cam.width = width
        cam.height = height
        cam.zoom = 1 - margin
        self.refresh()

    def refresh(self) -> None:
        self._canvas.update()
        self._canvas.request_draw(self._animate)

    def _animate(self) -> None:
        self._renderer.render(self._scene, self._camera)

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        # Code adapted from:
        # https://github.com/pygfx/pygfx/pull/753/files#diff-173d643434d575e67f8c0a5bf2d7ea9791e6e03a4e7a64aa5fa2cf4172af05cdR395
        viewport = pygfx.Viewport.from_viewport_or_renderer(self._renderer)
        if not viewport.is_inside(*pos_xy):
            return (-1, -1, -1)

        # Get position relative to viewport
        pos_rel = (
            pos_xy[0] - viewport.rect[0],
            pos_xy[1] - viewport.rect[1],
        )

        vs = viewport.logical_size

        # Convert position to NDC
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        if self._camera:
            pos_ndc += la.vec_transform(
                self._camera.world.position, self._camera.camera_matrix
            )
            pos_world = la.vec_unproject(pos_ndc[:2], self._camera.camera_matrix)

            # NB In vispy, (0.5,0.5) is a center of an image pixel, while in pygfx
            # (0,0) is the center. We conform to vispy's standard.
            return (pos_world[0] + 0.5, pos_world[1] + 0.5, pos_world[2] + 0.5)
        else:
            return (-1, -1, -1)
