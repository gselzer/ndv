from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np

    from ndv._views.bases import LutView
    from ndv._views.bases.graphics._canvas_elements import ImageHandle
    from ndv.models._lut_model import LUTModel

    LutKey = int | None


class ChannelController:
    """Controller for a single channel in the viewer.

    This manages the connection between the LUT model (settings like colormap,
    contrast limits and visibility) and the LUT view (the front-end widget that
    allows the user to interact with these settings), as well as the image handle
    that displays the data, all for a single "channel" extracted from the data.
    """

    def __init__(
        self, key: LutKey, lut_model: LUTModel, views: Sequence[LutView]
    ) -> None:
        self.key = key
        self.lut_views: list[LutView] = []
        self.lut_model = lut_model
        self.handles: list[ImageHandle] = []
        for v in views:
            self.add_lut_view(v)

    def add_lut_view(self, view: LutView) -> None:
        """Add a LUT view to the controller."""
        view.model = self.lut_model
        self.lut_views.append(view)
        self.synchronize(view)

    def synchronize(self, *views: LutView) -> None:
        """Make sure the view matches the model."""
        _views: Iterable[LutView] = views or self.lut_views
        name = str(self.key) if self.key is not None else ""
        for view in _views:
            view.synchronize()
            view.set_channel_name(name)

    def update_texture_data(self, data: np.ndarray) -> None:
        """Update the data in the image handle."""
        # WIP:
        # until we have a more sophisticated way to handle updating data
        # for multiple handles, we'll just update the first one
        if not (handles := self.handles):
            return
        handles[0].set_data(data)
        # if this image handle is visible and autoscale is on, then we need
        # to update the clim values
        if self.lut_model.autoscale:
            self.lut_model.clims = (data.min(), data.max())
            # lut_view.setClims((data.min(), data.max()))
            # technically... the LutView may also emit a signal that the
            # controller listens to, and then updates the image handle
            # but this next line is more direct
            # self._handles[None].clim = (data.min(), data.max())

    def add_handle(self, handle: ImageHandle) -> None:
        """Add an image texture handle to the controller."""
        handle.model = self.lut_model
        self.handles.append(handle)
        self.add_lut_view(handle)

        if self.lut_model.autoscale:
            data = handle.data()
            self.lut_model.clims = (data.min(), data.max())

    def get_value_at_index(self, idx: tuple[int, ...]) -> float | None:
        """Get the value of the data at the given index."""
        if not (handles := self.handles):
            return None
        # only getting one handle per channel for now
        handle = handles[0]
        with suppress(IndexError):  # skip out of bounds
            # here, we're retrieving the value from the in-memory data
            # stored by the backend visual, rather than querying the data itself
            # this is a quick workaround to get the value without having to
            # worry about other dimensions in the data source (since the
            # texture has already been reduced to 2D). But a more complete
            # implementation would gather the full current nD index and query
            # the data source directly.
            return handle.data()[idx]  # type: ignore [no-any-return]
        return None
