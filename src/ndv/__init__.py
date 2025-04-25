"""Fast and flexible n-dimensional data viewer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ndv")
except PackageNotFoundError:
    __version__ = "uninstalled"

from . import data
from .controllers import ArraysViewer, ArrayViewer
from .models import DataWrapper
from .util import imshow
from .views import run_app, set_canvas_backend, set_gui_backend

__all__ = [
    "ArrayViewer",
    "ArraysViewer",
    "DataWrapper",
    "data",
    "imshow",
    "run_app",
    "set_canvas_backend",
    "set_gui_backend",
]
