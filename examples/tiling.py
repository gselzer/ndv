from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import QApplication, QPushButton

from ndv import NDViewer

if TYPE_CHECKING:
    from qtpy.QtCore import QCoreApplication


def _get_app() -> tuple[QCoreApplication, bool]:
    is_ipython = False
    if (app := QApplication.instance()) is None:
        app = QApplication([])
        app.setApplicationName("ndv")
    elif (ipy := sys.modules.get("IPython")) and (shell := ipy.get_ipython()):
        is_ipython = str(shell.active_eventloop).startswith("qt")

    return app, not is_ipython


app, _ = _get_app()

viewer = NDViewer()

tiles = []
img_shape = (10, 10)
tile: np.ndarray
# FIXME? Unfortunately, but naturally, there's a performance drop when you
# add multiple datasets. Vispy drops about 3 FPS
for i in range(3):
    for j in range(3):
        tile = np.ones(img_shape) * (i + j)
        tile[0, 0] += 1
        tiles.append(tile)
        viewer.add_data(data=tile, position=(10 * i, 10 * j))


def randomize_last_tile() -> None:
    global tile
    global img_shape
    tile[:, :] = np.random.randint(low=1, high=5, size=img_shape)
    # FIXME: This doesn't actually update the image on the canvas
    # The fix may be present within https://github.com/pyapp-kit/ndv/pull/41
    viewer.set_current_index()


random_btn = QPushButton("Randomize!")
random_btn.clicked.connect(randomize_last_tile)
viewer._btns.addWidget(random_btn)

viewer.show()
viewer.raise_()

app.exec()
