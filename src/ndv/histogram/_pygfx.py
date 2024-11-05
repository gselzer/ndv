from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx
from fastplotlib.graphics import LineGraphic
from fastplotlib.graphics._base import Graphic
from fastplotlib.graphics.selectors import LinearRegionSelector
from fastplotlib.graphics.utils import pause_events

if TYPE_CHECKING:
    from typing import Any


class HistogramGraphic(Graphic):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._group = pygfx.Group()
        self._set_world_object(self._group)

        self._vmin = -10
        self._vmax = 10

        self._line = LineGraphic(np.zeros((22, 3)))
        self._group.add(self._line.world_object)
        self._clims = LinearRegionSelector(
            selection=[-5, 5],
            limits=[-10, 10],
            center=0,
            size=0,
            axis="x",
            edge_thickness=8,
            resizable=True,
            parent=self,
        )

        # there will be a small difference with the histogram edges so this makes
        # them both line up exactly
        self._clims.selection = (
            self._vmin,
            self._vmax,
        )
        # self._clims.add_event_handler(
        #     self._linear_region_handler, "selection"
        # )

        self._group.add(self._clims.world_object)

    def _create_line(self, n: int) -> None:
        if self._line:
            self._group.remove(self._line.world_object)

    def _linear_region_handler(self, ev):
        # must use world coordinate values directly from selection()
        # otherwise the linear region bounds jump to the closest bin edges
        selected_ixs = self._clims.selection
        vmin, vmax = selected_ixs[0], selected_ixs[1]
        vmin, vmax = vmin / self._scale_factor, vmax / self._scale_factor
        self.vmin, self.vmax = vmin, vmax

    def _fpl_add_plot_area_hook(self, plot_area):
        self._plot_area = plot_area
        self._clims._fpl_add_plot_area_hook(plot_area)
        self._line._fpl_add_plot_area_hook(plot_area)

        self._plot_area.auto_scale()
        self._plot_area.controller.enabled = True

    @property
    def vmin(self) -> float:
        return self._vmin

    @vmin.setter
    def vmin(self, value: float) -> None:
        with pause_events(self.image_graphic, self._clims):
            # must use world coordinate values directly from selection()
            # otherwise the linear region bounds jump to the closest bin edges
            self._clims.selection = (
                value * self._scale_factor,
                self._clims.selection[1],
            )
            self.image_graphic.vmin = value

        self._vmin = value
        if self._colorbar is not None:
            self._colorbar.vmin = value

        vmin_str, vmax_str = self._get_vmin_vmax_str()
        self._text_vmin.offset = (-120, self._clims.selection[0], 0)
        self._text_vmin.text = vmin_str

    @property
    def vmax(self) -> float:
        return self._vmax

    @vmax.setter
    def vmax(self, value: float) -> None:
        with pause_events(self.image_graphic, self._clims):
            # must use world coordinate values directly from selection()
            # otherwise the linear region bounds jump to the closest bin edges
            self._clims.selection = (
                self._clims.selection[0],
                value * self._scale_factor,
            )

            self.image_graphic.vmax = value

        self._vmax = value
        if self._colorbar is not None:
            self._colorbar.vmax = value

        vmin_str, vmax_str = self._get_vmin_vmax_str()
        self._text_vmax.offset = (-120, self._clims.selection[1], 0)
        self._text_vmax.text = vmax_str

    def set_data(self, values: np.ndarray, bin_edges: np.ndarray) -> None:
        self._line.data[:, 0] = np.repeat(bin_edges, 2)  # xs
        self._line.data[0, 1] = self._line.data[21, 1] = 0
        self._line.data[1:-1, 1] = np.repeat(values, 2)  # xs

        self._clims.limits = [bin_edges[0], bin_edges[-1]]
        height = values.max() * 0.98
        self._clims.fill.geometry = pygfx.box_geometry(1, height, 1)
        self._clims.edges[0].geometry.positions.data[:, 1] = [height / 2, -height / 2]
        self._clims.edges[1].geometry.positions.data[:, 1] = [height / 2, -height / 2]
        self._clims.offset = [0, height / 2, 0]


# class FastplotlibHistogramView(HistogramView):

#     def __init__(self) -> None:
#         self._wdg = HistogramLUT()
