"""Test controller without canavs or gui frontend"""

from __future__ import annotations

import gc
import os
import weakref
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, cast, no_type_check
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import scenex as snx
from app_model.types import KeyBinding, KeyCode, KeyMod, SimpleKeyBinding
from scenex.app.events import (
    KeyPressEvent,
    MouseButton,
    MouseLeaveEvent,
    MouseMoveEvent,
    WheelEvent,
)

from ndv.controllers import ArrayViewer
from ndv.controllers._channel_controller import ChannelController
from ndv.models import DataWrapper
from ndv.models._array_display_model import ArrayDisplayModel, ChannelMode
from ndv.models._lut_model import ClimsManual, ClimsMinMax, LUTModel
from ndv.models._resolve import DataResponse, resolve
from ndv.views import _app, gui_frontend
from ndv.views._histogram import Histogram
from ndv.views.bases import ArrayView, LUTView

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from qtpy import API_NAME
except ImportError:
    API_NAME = None

IS_WIN = os.name == "nt"
IS_PYSIDE6 = API_NAME == "PySide6"


def _get_mock_hist_canvas() -> Histogram:
    return MagicMock(spec=Histogram)


def _get_mock_view(*_: Any) -> ArrayView:
    mock = MagicMock(spec=ArrayView)
    mock.add_lut_view.side_effect = lambda *a, **k: MagicMock(spec=LUTView)
    return mock


def _patch_views(f: Callable) -> Callable:
    f = patch.object(_app, "get_array_view_class", lambda: _get_mock_view)(f)
    f = patch.object(_app, "filter_key_events", lambda *a, **k: lambda: None)(f)
    return f


@no_type_check
@_patch_views
def test_controller() -> None:
    SHAPE = (10, 4, 10, 10)
    ctrl = ArrayViewer()
    ctrl._async = False
    model = ctrl.display_model
    mock_view = ctrl._view
    mock_view.create_sliders.assert_not_called()

    data = np.empty(SHAPE)
    ctrl.data = data
    wrapper = ctrl._data_wrapper

    # showing the controller shows the view
    ctrl.show()
    mock_view.set_visible.assert_called_once_with(True)

    # sliders are first created with the shape of the data
    ranges = {i: range(s) for i, s in enumerate(SHAPE)}
    mock_view.create_sliders.assert_called_once_with(ranges)
    # visible-axis sliders are hidden
    # (2,3) because model.visible_axes is set to (-2, -1) and ndim is 4
    mock_view.hide_sliders.assert_called_once_with({2, 3}, show_remainder=True)
    # channel mode is set to default (which is currently grayscale)
    mock_view.set_channel_mode.assert_called_once_with(model.channel_mode)
    # data info is set
    mock_view.set_data_info.assert_called_once_with(wrapper.summary_info())
    model.current_index.assign({0: 1})

    # changing visible axes updates which sliders are visible
    model.visible_axes = (0, 3)
    mock_view.hide_sliders.assert_called_with({0, 3}, show_remainder=True)

    # changing the channel mode updates the sliders and updates the view combobox
    mock_view.hide_sliders.reset_mock()
    model.channel_mode = "composite"
    mock_view.set_channel_mode.assert_called_with(ChannelMode.COMPOSITE)
    # axis 1 is guessed as channel_axis by resolve() (not stored on model)
    mock_view.hide_sliders.assert_called_once_with({0, 1, 3}, show_remainder=True)
    model.channel_mode = ChannelMode.GRAYSCALE
    mock_view.hide_sliders.assert_called_with({0, 3}, show_remainder=True)

    # when the view changes the current index, the model is updated
    idx = {0: 1, 1: 2, 3: 8}
    mock_view.current_index.return_value = idx
    ctrl._on_view_current_index_changed()
    assert model.current_index == idx

    # when the view requests 3 dimensions, the model is updated
    ctrl._on_view_ndim_toggle_requested(True)
    assert len(model.visible_axes) == 3

    # when the view changes the channel mode, the model is updated
    assert model.channel_mode == ChannelMode.GRAYSCALE
    ctrl._on_view_channel_mode_changed(ChannelMode.COMPOSITE)
    assert model.channel_mode == ChannelMode.COMPOSITE

    # setting a new ArrayDisplay model updates the appropriate view widgets
    ch_ctrl = cast("ChannelController", ctrl._lut_controllers[None])
    ch_ctrl.lut_views[0].set_colormap.reset_mock()
    ctrl.display_model = ArrayDisplayModel(default_lut=LUTModel(cmap="green"))
    # fails
    # ch_ctrl.lut_views[0].set_colormap_without_signal.assert_called_once()


@no_type_check
@_patch_views
def test_canvas_interaction() -> None:
    SHAPE = (10, 4, 10, 10)
    data = np.zeros(SHAPE)
    ctrl = ArrayViewer()
    ctrl._async = False
    canvas = ctrl._canvas

    mock_view = ctrl._view
    ctrl.data = data

    ctrl._add_histogram(None)
    histogram = ctrl._histograms[None]

    # clicking the reset zoom button calls reset_zoom on the canvas
    tform_before = canvas.view.camera.transform
    canvas.view.camera.transform = tform_before.translated([1e6, 1e6, 1e6])
    ctrl._on_view_reset_zoom_clicked()
    assert tform_before == canvas.view.camera.transform

    # hovering on the image updates the hover info in the view
    mock_view.reset_mock()
    pos = (canvas._canvas.width // 2, canvas._canvas.height // 2)
    ctrl._view_event(MouseMoveEvent(pos=pos, buttons=MouseButton.NONE))
    world_pos = canvas.view.to_ray(pos).origin[:2]
    y, x = int(world_pos[1]), int(world_pos[0])
    mock_view.set_hover_info.assert_called_once_with(f"[{y}, {x}] 0")
    assert histogram.highlight_line.transform.root[3, 0] == 0

    mock_view.reset_mock()
    # updating the image also updates the hover info in the view
    # NB Since the image handle is a mock, the data won't be updated.
    ctrl.data = np.ones(SHAPE, dtype=np.uint8)
    # FIXME: These methods are actually called twice, both within
    # _fully_synchronize_view. The first time is on
    # ArrayViewer._on_view_current_index_change, and the second on
    # ArrayViewer._request_data
    mock_view.set_hover_info.assert_called_once_with(f"[{y}, {x}] 1")
    assert histogram.highlight_line.transform.root[3, 0] == 1

    mock_view.reset_mock()

    # hovering off the image clears the hover info in the view
    ctrl._view_event(MouseMoveEvent(pos=(0, 0), buttons=MouseButton.NONE))
    mock_view.set_hover_info.assert_called_once_with("")
    assert not histogram.highlight_line.visible

    mock_view.reset_mock()

    # leaving the canvas clears the hover info as well
    ctrl._view_event(MouseLeaveEvent())
    mock_view.set_hover_info.assert_called_once_with("")
    assert not histogram.highlight_line.visible


@no_type_check
@_patch_views
@patch("ndv.controllers._array_viewer.Histogram", _get_mock_hist_canvas)
def test_histogram_controller() -> None:
    ctrl = ArrayViewer()
    ctrl._async = False
    mock_view = ctrl._view

    ctrl.data = np.zeros((10, 4, 10, 10)).astype(np.uint8)

    # adding a histogram tells the view to add a histogram, and updates the data
    ctrl._add_histogram(None)
    mock_view.add_histogram.assert_called_once()
    mock_histogram = ctrl._histograms[None]
    mock_histogram.set_data.assert_called_once()

    # changing the index updates the histogram
    mock_histogram.set_data.reset_mock()
    ctrl.display_model.current_index.assign({0: 1, 1: 2, 3: 3})
    mock_histogram.set_data.assert_called_once()

    # switching to composite mode puts the histogram view in the
    # lut controller for all channels (this may change)
    ctrl.display_model.channel_mode = ChannelMode.COMPOSITE
    assert mock_histogram in ctrl._lut_controllers[None].lut_views


@no_type_check
@_patch_views
def test_histogram_updates_on_first_draw() -> None:
    """Histogram should update even when the first response creates the handle."""
    ctrl = ArrayViewer()

    ctrl._histograms[None] = hist = MagicMock(spec=Histogram)
    ctrl._lut_controllers[None] = ChannelController(
        key=None, lut_model=LUTModel(), views=[MagicMock(spec=LUTView), hist]
    )

    response = DataResponse(
        n_visible_axes=2,
        data={None: np.arange(100, dtype=np.uint8).reshape(10, 10)},
    )
    future: Future[DataResponse] = Future()
    future.set_result(response)
    ctrl._futures[future] = ctrl._current_gen

    ctrl._on_data_response_ready(future)

    hist.set_data.assert_called_once()


@pytest.mark.usefixtures("any_app")
def test_array_viewer_with_app() -> None:
    """Example usage of new mvc pattern."""
    viewer = ArrayViewer()
    assert gui_frontend() in type(viewer._view).__name__.lower()
    viewer.show()

    data = np.random.randint(0, 255, size=(10, 10, 10, 10, 10), dtype="uint8")
    viewer.data = data

    # test changing current index via the view
    index_mock = Mock()
    viewer.display_model.current_index.value_changed.connect(index_mock)
    index = {0: 4, 1: 1, 2: 2}
    # setting the index should trigger the signal, only once
    viewer._view.set_current_index(index)
    index_mock.assert_called_once()
    for k, v in index.items():
        assert viewer.display_model.current_index[k] == v

    # setting again should not trigger the signal
    index_mock.reset_mock()
    viewer._view.set_current_index(index)
    index_mock.assert_not_called()

    # test_setting 3D via model
    assert viewer.display_model.visible_axes == (-2, -1)
    visax_mock = Mock()
    viewer.display_model.events.visible_axes.connect(visax_mock)
    viewer.display_model.visible_axes = (0, -2, -1)
    visax_mock.assert_called_once()
    assert viewer.display_model.visible_axes == (0, -2, -1)


@pytest.mark.usefixtures("any_app")
def test_channel_autoscale() -> None:
    ctrl = ChannelController(key=None, lut_model=LUTModel(), views=[])

    # NB: Use a planar dataset so we can manually compute the min/max
    data = np.random.randint(0, 255, size=(10, 10), dtype="uint8")
    mi, ma = np.nanmin(data), np.nanmax(data)
    handle = MagicMock(spec=snx.Image)
    handle.data = data
    ctrl.add_image(handle)

    # Test some random LutController
    lut_model = ctrl.lut_model
    lut_model.clims = ClimsManual(min=1, max=2)

    # Ensure newly added lut views have the correct clims
    mock_viewer = MagicMock(LUTView)
    ctrl.add_lut_view(mock_viewer)
    mock_viewer.set_clims.assert_called_once_with((1, 2))

    # Ensure autoscaling sets the clims
    mock_viewer.set_clims.reset_mock()
    lut_model.clims = ClimsMinMax()
    mock_viewer.set_clims.assert_called_once_with((mi, ma))


@pytest.mark.usefixtures("any_app")
def test_array_viewer_histogram() -> None:
    """Mostly a smoke test for basic functionality of histogram backends."""

    viewer = ArrayViewer()
    viewer.show()
    viewer._add_histogram(None)
    histogram = viewer._histograms.get(None, None)
    assert histogram is not None

    # change views
    histogram.set_log_base(10)

    # update data
    np.random.seed(0)
    maxval = 2**16 - 1
    data = np.random.randint(0, maxval, (1000,), dtype="uint16")
    counts = np.bincount(data.flatten(), minlength=maxval + 1)
    bin_edges = np.arange(maxval + 2) - 0.5
    histogram.set_data(counts, bin_edges)


@pytest.mark.allow_leaks
@pytest.mark.usefixtures("any_app")
def test_rgb_display_magic() -> None:
    # FIXME: Something in the QLUTView is causing leaked qt widgets here.
    # Doesn't seem to be coming from the QRGBView...
    def assert_rgb_magic_works(rgb_data: np.ndarray) -> None:
        viewer = ArrayViewer(rgb_data)
        assert viewer.display_model.channel_mode == ChannelMode.RGBA
        # Note Multiple correct answers here - modulus covers both cases
        assert cast("int", viewer.display_model.channel_axis) % rgb_data.ndim == 4
        assert cast("int", viewer.display_model.visible_axes[0]) % rgb_data.ndim == 2
        assert cast("int", viewer.display_model.visible_axes[1]) % rgb_data.ndim == 3

    rgb_data = np.ones((1, 2, 3, 4, 3), dtype=np.uint8)
    assert_rgb_magic_works(rgb_data)

    rgba_data = np.ones((1, 2, 3, 4, 4), dtype=np.uint8)
    assert_rgb_magic_works(rgba_data)


def test_resolve_is_pure() -> None:
    """Test that resolve() does not mutate the input model."""
    wrapper = DataWrapper.create(np.empty((2, 3, 4, 5)))
    model = ArrayDisplayModel()
    model.current_index.assign({0: 7, -4: 1})
    before = dict(model.current_index)

    resolved = resolve(model, wrapper)

    # model should be untouched
    assert dict(model.current_index) == before
    # resolved normalizes -4 → 0, so duplicate key picked the non-int value
    assert resolved.current_index[0] == 1


@no_type_check
@_patch_views
def test_stale_response_discard() -> None:
    """Test that responses from old request generations are discarded."""
    ctrl = ArrayViewer()

    # simulate a stale response from generation 1 when we're on generation 2
    future: Future[DataResponse] = Future()
    future.result = Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("result() must not be called on stale future")
    )
    ctrl._current_gen = 2
    ctrl._futures[future] = 1  # old generation

    ctrl._on_data_response_ready(future)

    # stale response should be ignored — no LUT controllers created, no images
    assert len(ctrl._lut_controllers) == 0
    scene = ctrl._canvas.view.scene
    data_nodes = [node for node in scene.children if isinstance(node, snx.Image)]
    assert len(data_nodes) == 0


@no_type_check
@_patch_views
def test_rgba_3d_fallback_warns() -> None:
    """Test that RGBA mode with 3D view reverts to GRAYSCALE with a warning."""
    ctrl = ArrayViewer(np.zeros((10, 4, 10, 10)))
    ctrl.display_model.visible_axes = (0, 2, 3)

    with pytest.warns(UserWarning, match="Cannot use RGBA mode with 3D view"):
        ctrl.display_model.channel_mode = ChannelMode.RGBA

    assert ctrl.display_model.channel_mode == ChannelMode.GRAYSCALE


@no_type_check
@_patch_views
def test_set_scales_called_on_apply() -> None:
    """set_scales is called on the canvas when scales change."""
    ctrl = ArrayViewer(np.empty((3, 100, 200)))
    ctrl._async = False

    with patch.object(ctrl._canvas, "set_scales") as mock_set_scales:
        ctrl.display_model.scales[1] = 0.5
        mock_set_scales.assert_called()
        args = mock_set_scales.call_args[0][0]
        assert args[0] == 0.5  # axis 1


@no_type_check
def test_fallback_channel_names_pushed() -> None:
    """Fallback channel names are pushed to LUT views."""
    ctrl = ArrayViewer(
        display_model=ArrayDisplayModel(
            channel_axis=0,
            channel_mode="composite",
        ),
    )
    ctrl._async = False
    ctrl.data = np.empty((3, 10, 10))

    # fallback names default to str(key) for plain numpy arrays
    for key, lut_ctrl in ctrl._lut_controllers.items():
        if isinstance(key, int):
            for view in lut_ctrl.lut_views:
                assert view._fallback_name == str(key)
                # assert view.set_fallback_name.assert_called_with(str(key))


@no_type_check
@_patch_views
def test_scales_applied_after_async_data_response() -> None:
    """Scales must be applied after handles are created in async data response.

    Regression: set_scales() in _apply_changes() runs before handles exist in
    async mode, so first paint is unscaled unless scales are re-applied after
    handle creation in _on_data_response_ready().
    """
    ctrl = ArrayViewer(display_model=ArrayDisplayModel(scales={-2: 0.5, -1: 2.0}))

    # Set up data wrapper and resolve state without triggering data fetch
    ctrl._data_wrapper = DataWrapper.create(np.empty((10, 100, 200)))
    ctrl._resolved = resolve(ctrl._display_model, ctrl._data_wrapper)

    with patch.object(ctrl._canvas, "set_scales") as mock_set_scales:
        # Simulate the async data response arriving (creates handles)
        response = DataResponse(
            n_visible_axes=2, data={None: np.zeros((100, 200), dtype=np.uint8)}
        )
        future: Future[DataResponse] = Future()
        future.set_result(response)
        ctrl._futures[future] = ctrl._current_gen
        ctrl._on_data_response_ready(future)

    # set_scales must be called AFTER handles are created
    mock_set_scales.assert_called()
    last_scales = mock_set_scales.call_args[0][0]
    assert last_scales == (0.5, 2.0)


@no_type_check
@_patch_views
def test_data_replacement_with_stale_index() -> None:
    """Replacing data with fewer dims should not crash due to stale current_index."""
    # Start with 4D data — axes 0, 1, 2, 3 are all valid
    ctrl = ArrayViewer(np.empty((10, 3, 64, 64)))
    ctrl._async = False

    # Set an index on axis 3 (valid for 4D data)
    ctrl.display_model.current_index[3] = 1

    # Replace with 3D data — axis 3 no longer exists.
    # _norm_current_index calls normalize_axis_key(3) which will raise IndexError
    # because the new data only has 3 dims (valid axes: 0, 1, 2).
    ctrl.data = np.empty((20, 32, 32))

    # Should not have raised. The viewer should be in a usable state.
    ctrl._request_data()
    ctrl._join()


@no_type_check
@_patch_views
def test_remove_lut_view_with_non_gui_view() -> None:
    """remove_lut_view should handle non-GUI LUTViews (e.g. ImageHandle).

    See https://github.com/pyapp-kit/ndv/issues/138
    """
    ctrl = ArrayViewer()
    ctrl._async = False
    ctrl.data = np.random.randint(0, 255, size=(10, 10), dtype="uint8")

    lut_ctrl = next(iter(ctrl._lut_controllers.values()))
    # lut_views contains both the GUI LUTView and the ImageHandle
    for view in lut_ctrl.lut_views:
        # This should not raise, even for non-GUI views like ImageHandle
        ctrl._view.remove_lut_view(view)


@no_type_check
@_patch_views
def test_user_current_index_preserved_on_init() -> None:
    """User-provided current_index must not be overwritten by slider defaults."""
    user_index = {0: 5}
    ctrl = ArrayViewer(current_index=user_index)
    ctrl._view.current_index.return_value = {0: 0, 1: 0}
    ctrl._async = False
    ctrl.data = np.empty((10, 3, 64, 64))
    assert ctrl.display_model.current_index[0] == 5


@no_type_check
@_patch_views
def test_keybinding_slice_navigation() -> None:
    """Arrow keys step focused slider and cycle focused axis."""
    from ndv._keybindings import _ensure_focused_axis

    SHAPE = (5, 10, 128, 128)
    ctrl = ArrayViewer()
    ctrl._async = False
    ctrl._view.current_index.return_value = {0: 0, 1: 0}
    ctrl.data = np.empty(SHAPE)

    # visible_axes should be (-2, -1) -> (2, 3) for 4D data
    # steppable axes are 0 and 1
    # default focused axis is the last steppable axis (1)
    assert ctrl._focused_slider_axis is None  # not yet set
    assert _ensure_focused_axis(ctrl) == 1
    assert ctrl._focused_slider_axis == 1

    def press(key: KeyCode | str, mods: KeyMod = KeyMod.NONE) -> None:
        kb = KeyBinding(parts=[SimpleKeyBinding(key=key, mods=mods)])
        ctrl._view_event(KeyPressEvent(key=kb))

    # RIGHT arrow steps forward on focused axis (1)
    press(KeyCode.RightArrow)
    assert ctrl.display_model.current_index[1] == 1

    # Another RIGHT
    press(KeyCode.RightArrow)
    assert ctrl.display_model.current_index[1] == 2

    # LEFT arrow steps backward
    press(KeyCode.LeftArrow)
    assert ctrl.display_model.current_index[1] == 1

    # UP arrow cycles to previous axis (0)
    press(KeyCode.UpArrow)
    assert ctrl._focused_slider_axis == 0

    # RIGHT now steps axis 0
    press(KeyCode.RightArrow)
    assert ctrl.display_model.current_index[0] == 1

    # DOWN cycles back to axis 1
    press(KeyCode.DownArrow)
    assert ctrl._focused_slider_axis == 1

    # LEFT doesn't go below 0
    ctrl.display_model.current_index[1] = 0
    press(KeyCode.LeftArrow)
    assert ctrl.display_model.current_index[1] == 0

    # RIGHT doesn't go above max
    ctrl.display_model.current_index[1] = 9  # max for shape 10
    press(KeyCode.RightArrow)
    assert ctrl.display_model.current_index[1] == 9

    # Unrecognized key does nothing
    press("x")
    assert ctrl.display_model.current_index[1] == 9


@no_type_check
@_patch_views
def test_keybinding_zoom() -> None:
    """Plus/minus keys should call canvas.zoom when mouse is over canvas."""

    ctrl = ArrayViewer()
    ctrl._async = False
    ctrl.data = np.empty((10, 10))
    canvas_pos = (ctrl._canvas._canvas.width // 2, ctrl._canvas._canvas.height // 2)

    def press(key: str, mods: KeyMod = KeyMod.NONE) -> None:
        kb = KeyBinding(parts=[SimpleKeyBinding(key=key, mods=mods)])
        ctrl._view_event(KeyPressEvent(key=kb))

    mouse_in = WheelEvent(canvas_pos, MouseButton.NONE, angle_delta=(0, 120))
    mouse_out = WheelEvent(canvas_pos, MouseButton.NONE, angle_delta=(0, -120))

    # = key (zoom in)
    with patch.object(snx.PanZoom, "handle_event") as mock_handle_event:
        press("=")
    mock_handle_event.assert_called_once_with(mouse_in, ctrl._canvas.view)

    # - key (zoom out)
    with patch.object(snx.PanZoom, "handle_event") as mock_handle_event:
        press("-")
    mock_handle_event.assert_called_once_with(mouse_out, ctrl._canvas.view)

    # + (shift+=) should also zoom in
    with patch.object(snx.PanZoom, "handle_event") as mock_handle_event:
        press("=", KeyMod.Shift)
    mock_handle_event.assert_called_once_with(mouse_in, ctrl._canvas.view)

    # _ (shift+-) should also zoom out
    with patch.object(snx.PanZoom, "handle_event") as mock_handle_event:
        press("-", KeyMod.Shift)
    mock_handle_event.assert_called_once_with(mouse_out, ctrl._canvas.view)


@no_type_check
@pytest.mark.usefixtures("any_app")
def test_handle_gc_on_data_reassign() -> None:
    """Image handles should be GC'd when viewer.data is reassigned."""
    viewer = ArrayViewer()
    viewer._async = False
    viewer.data = np.zeros((10, 10), dtype="uint8")

    ctrl = next(iter(viewer._lut_controllers.values()))
    handle_ref = weakref.ref(ctrl.handles[0])

    viewer.data = np.zeros((10, 10), dtype="uint8")
    gc.collect()

    assert handle_ref() is None
