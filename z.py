# test_example = true
from __future__ import annotations

import fastplotlib as fpl
import numpy as np
from fastplotlib.layouts import Figure
from pygfx.controllers import PanZoomController

from ndv.histogram._pygfx import HistogramGraphic


class PanZoom1DController(PanZoomController):
    """A PanZoomController that locks one axis."""

    _zeros = np.zeros(3)

    def _update_pan(self, delta: tuple, *, vecx, vecy) -> None:
        super()._update_pan(delta, vecx=vecx, vecy=self._zeros)


figure = Figure(
    size=(700, 560),
)

data = np.random.normal(10, 10, 10000)
values, bin_edges = np.histogram(data)
h = HistogramGraphic()
h.set_data(values, bin_edges)
g = figure[0, 0].add_graphic(h)
figure[0, 0].camera.maintain_aspect = False
figure[0, 0].controller = PanZoom1DController()
figure.show()

# NOTE: `if __name__ == "__main__"` is NOT how to use fastplotlib interactively
# please see our docs for using fastplotlib interactively in ipython and jupyter
if __name__ == "__main__":
    print(__doc__)
    fpl.run()
