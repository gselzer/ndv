# ndv

[![License](https://img.shields.io/pypi/l/ndv.svg?color=green)](https://github.com/pyapp-kit/ndv/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ndv.svg?color=green)](https://pypi.org/project/ndv)
[![Python Version](https://img.shields.io/pypi/pyversions/ndv.svg?color=green)](https://python.org)
[![CI](https://github.com/pyapp-kit/ndv/actions/workflows/ci.yml/badge.svg)](https://github.com/pyapp-kit/ndv/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pyapp-kit/ndv/branch/main/graph/badge.svg)](https://codecov.io/gh/pyapp-kit/ndv)

Simple, fast-loading, asynchronous, n-dimensional array viewer for Qt, with minimal dependencies.

```python
import ndv

data = ndv.data.cells3d()
# or ndv.data.nd_sine_wave()
# or *any* arraylike object (see support below)

ndv.imshow(data)
```

![Montage](https://github.com/pyapp-kit/ndv/assets/1609449/712861f7-ddcb-4ecd-9a4c-ba5f0cc1ee2c)

As an alternative to `ndv.imshow()`, you can instantiate the `ndv.NDViewer` (`QWidget` subclass) directly

```python
from qtpy.QtWidgets import QApplication
from ndv import NDViewer

app = QApplication([])
viewer = NDViewer(data)
viewer.show()
app.exec()
```

## Features

<!-- Sounds really boring - can we spice it up? -->
- data views on a 2D/3D canvas
- sliders for all non-visible dims, supporting integer as well as slice (range)-based slicing
- colormaps provided by [cmap](https://github.com/tlambert03/cmap)

## Installation

The only required dependencies are `numpy` and `superqt[cmap,iconify]`.
You will also need a Qt backend (PyQt or PySide) and one of either
[vispy](https://github.com/vispy/vispy) or [pygfx](https://github.com/pygfx/pygfx),
which can be installed through extras `ndv[<pyqt|pyside>,<vispy|pygfx>]`:

```python
pip install ndv[pyqt,vispy]
```

> [!TIP]
> If you have both vispy and pygfx installed, `ndv` will default to using vispy,
> but you can override this with the environment variable
> `NDV_CANVAS_BACKEND=pygfx` or `NDV_CANVAS_BACKEND=vispy`

## Goals
<!-- In many ways, this should be treated as a roadmap -->

ndv provides data **visualization** and **inspection**. *data* in this case refers to an array of values which might be:
  - n-dimensional
  - updated (overwritten and/or appended)
  <!--TODO: Preserved these from Motivation - but can we flesh them out more?-->
  - asynchronously loaded
  - remote

To perform this task pleasantly, ndv strives to be:

- **General**: `ndv.NDViewer` supports (*but does not depend on*):
  - many **Canvas Backends**:
    - [vispy](https://github.com/vispy/vispy)
    - [pygfx](https://github.com/pygfx/pygfx)
  <!-- TODO: Widget frontends? -->
  - many **Widget Backends**:
    <!-- TODO: Better link? -->
    - [Qt](https://www.qt.io/)
    <!-- TODO: Actually support this? -->
    - [Jupyter](https://jupyter.org/)
  - many **Array Types**:
    - `numpy.ndarray`
    - `cupy.ndarray`
    - `dask.array.Array`
    - `jax.Array`
    - `pyopencl.array.Array`
    - `sparse.COO`
    - `tensorstore.TensorStore` (supports named dimensions)
    - `torch.Tensor` (supports named dimensions)
    - `xarray.DataArray` (supports named dimensions)
    - `zarr` (supports named dimensions)

    See examples for each of these array types in [examples](./examples/)

    <!-- TODO: Is this necessary? -->
    > [!NOTE]
    > *you can add support for any custom storage class by subclassing `ndv.datawrapper`
    > and implementing a couple methods.  
    > (this doesn't require modifying ndv, but contributions of new wrappers are welcome!)*


- **Snappy**: Minimal dependencies ensure quick import and load. `ndv.NDViewer` should be capbable of displaying "large" datasets at "high" framerates.

- **Metadata-aware**: `ndv.NDViewer` utilizes any provided metadata to the best of its ability, including:
    <!-- TODO: What else? -->
  - axis, channel labels
  - voxel scales

## Limitations

ndv avoids the following features, which would introduce scope creep and complicate the pursuit of our [goals](#goals):

- Plugins and/or Plugin Hooks
- Data Processing<sup>1</sup>
- Annotations & Segmentation<sup>1</sup>

> [!NOTE]
>
> <sup>1</sup> These are examples of *generative* tooling, where new things are created. Examples include a segmentation or a filtering. In opposition, ndv *should* include "inspective" tooling, which provides additional understanding of (regions of) the dataset. Inspective tooling that ndv has includes Region Selectors, Histograms, Line Profiles, and Lookup Tables.

## Alternatives

There are many existing projects that perform similar functions, with different focuses. We list popular alternatives below:

<!-- I do not know of another viewer that focuses on CHANGING data - could be a limitation of each -->

* [napari](https://github.com/napari/napari) provides interactive data visualization and acts as a unifying platform for a host of third-party plugins.
  * In practice, we found that napari required too many dependencies for the snappy startup we desired. It also neglects metadata, is tied into the VisPy/Qt pairing of backends, and does not handle changing datasets.
<!-- Is this event worth mentioning? -->
* [pyimagej](https://github.com/imagej/pyimagej) is a python wrapper around Fiji/ImageJ, providing data visualization, data processing and a host of third party plugins.
  * This library faces many of the same issues as napari, with the additional dependency of the JVM slowing visualization down further.
* [fastplotlib](https://github.com/fastplotlib/fastplotlib) provides plotting tools for data visualization.
  * This library is tied to pygfx, and is more concerned with data plotting.