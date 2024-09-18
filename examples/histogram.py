from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Ellipse, Line, Mesh, Polygon, Rectangle
from vispy.visuals.transforms import STTransform

if TYPE_CHECKING:
    from vispy.scene.events import SceneMouseEvent

data = np.zeros((10, 10), dtype=np.uint8)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i, j] = i + j


class GammaHandle(Ellipse, Movable):
    def __init__(self, curve: GammaCurve, *args: Any, **kwargs: Any) -> None:
        self.curve = curve
        Ellipse.__init__(
            self,
            center=[0.5, 0],
            radius=0.05,
            parent=curve,
            *args,
            **kwargs
        )
        self.interactive = True

    def start_move(self, pos: tuple[float, float]) -> None:
        self.curve.start_move(pos)

    def move(self, pos: tuple[float, float]) -> None:
        self.curve.move(pos)



class GammaCurve(scene.visuals.Compound):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__([], *args, **kwargs)

        self.unfreeze()

        # Magic number drawn from the number of points vispy
        # uses to approximate a bezier curve of points
        # [0, 0], [1, 1] with control point [0, 1]
        n = 14
        self._gamma = 1.
        self.color = "red"
        self._gamma_points: np.ndarray = np.ndarray((n, 2))
        self._gamma_points[:, 0] = np.linspace(0, 1, n)
        self._gamma_points[:, 1] = self._gamma_points[:, 0] ** self._gamma
        self._line = Line(pos=self._gamma_points, color=self.color, parent=self)

        self._handle = GammaHandle(curve=self, color=self.color)
        self._handle.transform = STTransform(translate=(0, 0.5 ** self._gamma))
        self.transform = STTransform()
        self._height_offset = 0
        
        self.freeze()

    def start_move(self, pos: tuple[float, float]) -> None:
        self._height_offset = pos[1] - self._handle.transform.translate[1]

    def _update_curve(self) -> None:
        self._gamma_points[:, 1] = self._gamma_points[:, 0] ** self._gamma
        self._line.set_data(pos=self._gamma_points)

    def move(self, pos: tuple[float, float]) -> None:
        p = pos[1] - self._height_offset
        #
        p_min = 0.0009765625
        p_max = 0.9330329915368074
        p = max(p_min, min(p_max, p))
        self._gamma = math.log(p, 0.5)
        self._handle.transform.translate = (0, p)
        self._update_curve()
    
    def _move_left(self, pos: float) -> None:
        ht = self._handle.transform.translate
        ht[0] = 


# TODO: Create Visual?
class LutEditor(scene.visuals.Compound):
    def __init__(self, data: np.ndarray, *args: Any, **kwargs: Any) -> None:
        super().__init__([], *args, **kwargs)

        self.unfreeze()

        self._bg = Rectangle(
            center=[0.5, 0.5], width=1, height=1, color="black", parent=self
        )
        self._bg.order = 10
        self._mesh = Histogram(data, parent=self)
        self._l_bound = Bound(parent=self, x=0, left=True)
        self._l_bound.order = -1
        self._r_bound = Bound(parent=self, x=1, left=False)

        self._gamma = GammaCurve(parent=self)

        self._dtype = data.dtype
        self.freeze()
    
    @property
    def clims(self) -> tuple[float, float]:
        range = self._mesh.range
        mi, ma = self._l_bound.value, self._r_bound.value
        # map [0, 1] onto [range[0], range[1]]
        return (
            mi * (range[1] - range[0]) + range[0],
            ma * (range[1] - range[0]) + range[0]
        )

    @clims.setter
    def clims(self, c: tuple[float, float]) -> None:
        range = self._mesh.range
        # map [range[0], range[1]] onto [0, 1]
        mi = c[0] - range[0] / (range[1] - range[0])
        ma = c[1] - range[0] / (range[1] - range[0])
        self._l_bound.value, self._r_bound.value = mi, ma
    
    @property
    def gamma(self) -> float:
        return self._gamma._gamma

    @gamma.setter
    def gamma(self, g: float) -> None:
        self._gamma._gamma = g


class Histogram(Mesh):
    def __init__(
        self,
        data: np.ndarray | None = None,
        color: Any = "blue",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, *args, **kwargs)

        self.unfreeze()
        self.max: float = 1
        self.bins = 256
        self.transform = STTransform()
        self.range: tuple[float, float] = (0, 1)
        self._data: np.ndarray | None = None
        if data is not None:
            self.data = data
            self._update_histogram()

        self.freeze()
    
    @property
    def data(self) -> np.ndarray:
        if self._data is not None:
            return self._data
        raise Exception("Data not yet set")
    
    @data.setter
    def data(self, d: np.ndarray) -> None:
        self._data = d
        try:
            iinfo = np.iinfo(data.dtype)
            self.range = (iinfo.min, iinfo.max)
        except Exception:
            raise Exception(f"Unsupported dtype: {data.dtype}")
        self._update_histogram()

        range = (self.range[1] - self.range[0])
        self.transform.scale = (1 / range, 1 / self.max)
        self.transform.translate = (-self.range[0] / range, 0)

    def _update_histogram(self) -> None:
        """Graciously adapted from vispy.visuals.histogram.py"""
        #   4-5
        #   | |
        # 1-2/7-8
        # |/| | |
        # 0-3-6-9
        #

        # do the histogramming
        data, bin_edges = np.histogram(self._data, self.bins, range=self.range)
        self.max = np.max(data)
        # construct our vertices
        rr = np.zeros((3 * len(bin_edges) - 2, 3), np.float32)
        rr[:, 0] = np.repeat(bin_edges, 3)[1:-1]
        rr[1::3, 1] = data
        rr[2::3, 1] = data
        bin_edges.astype(np.float32)
        # and now our tris
        tris = np.zeros((2 * len(bin_edges) - 2, 3), np.uint32)
        offsets = 3 * np.arange(len(bin_edges) - 1, dtype=np.uint32)[:, np.newaxis]
        tri_1 = np.array([0, 2, 1])
        tri_2 = np.array([2, 0, 3])
        tris[::2] = tri_1 + offsets
        tris[1::2] = tri_2 + offsets

        self.set_data(vertices=rr, faces=tris)


class LutEditorCanvas(SceneCanvas):
    def __init__(self) -> None:
        super().__init__(show=True, size=(200, 100))

        self.unfreeze()
        central_wdg: scene.Widget = self.central_widget
        self._view: scene.ViewBox = central_wdg.add_view()
        self._view.bgcolor = "darkgray"

        self.l = LutEditor(
            data=data,
            parent=self._view.scene,
        )
        self.l.transform = STTransform()
        camera = scene.PanZoomCamera(rect=(-0.1, -0.1, 1.2, 1.2), aspect=1)
        self._view.camera = camera
        camera.interactive = False
        self._on_mouse_move: list[Callable[[tuple[int, int]], None]] = []

        self.freeze()
    
    def on_mouse_press(self, event: SceneMouseEvent) -> None:
        pos = self._view.scene.transform.imap(event.pos)[:2]
        for v in self.visuals_at(event.pos):
            if isinstance(v, Movable):
                v.start_move(pos)
                self._on_mouse_move.append(v.move)
                return

    def on_mouse_move(self, event: SceneMouseEvent) -> None:
        pos = self._view.scene.transform.imap(event.pos)[:2]
        for f in self._on_mouse_move:
            f(pos)

    def on_mouse_release(self, event: SceneMouseEvent) -> None:
        self._on_mouse_move.clear()


canvas = LutEditorCanvas()

if __name__ == "__main__":
    if sys.flags.interactive != 1:
        app.run()
