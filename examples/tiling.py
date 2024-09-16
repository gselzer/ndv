from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QApplication

from ndv import NDViewer
from ndv.data import cells3d

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

viewer.set_data(cells3d(), position=(-10, -10))

viewer.show()
viewer.raise_()

app.exec()
