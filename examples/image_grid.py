# /// script
# dependencies = [
#     "ndv[pyqt,vispy]",
#     "imageio[tifffile]",
# ]
# ///
from __future__ import annotations

import os

import ndv

os.environ["SCENEX_CANVAS_BACKEND"] = "pygfx"
viewer = ndv.ArrayViewer()
viewer._async = False

data = ndv.data.cells3d()
for i in range(4):
    for j in range(3):
        img = viewer.add(data)
        img.transform = img.transform.translated([i * 266, j * 266, 0])

viewer.show()
ndv.run_app()
