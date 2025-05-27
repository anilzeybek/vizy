"""
vizy - lightweight tensor visualisation helper.

Install
-------
pip install vizy   # distribution name
import vizy

API
---
vizy.plot(tensor, **imshow_kwargs)  # show tensor as image or grid
vizy.save(path_or_tensor, tensor=None, **imshow_kwargs)  # save to file

If *tensor* is 4-D we assume shape is either (B, C, H, W) or (C, B, H, W) with C in {1,3}.
For ndarray/tensors of 2-D or 3-D we transpose to (H, W, C) as expected by Matplotlib.
"""

import math
import os
import tempfile
from typing import Any, Sequence

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

__all__: Sequence[str] = ("plot", "save")
__version__: str = "0.1.0"


def _to_numpy(x: Any) -> np.ndarray:
    """Convert *x* to NumPy array, detaching from torch if needed."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if not isinstance(x, np.ndarray):
        raise TypeError("Expected torch.Tensor | np.ndarray")
    return x


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """Ensure array is HxW or HxWxC where C in {1,3}."""
    if arr.ndim == 2:  # already HxW
        return arr
    if arr.ndim == 3:
        # if channels first
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        return arr
    raise ValueError(f"Unsupported dimensionality for _to_hwc: {arr.shape}")


def _prep(arr: np.ndarray) -> np.ndarray:
    arr = arr.squeeze()
    if arr.ndim in (2, 3):
        return _to_hwc(arr)
    if arr.ndim == 4:
        # try B,C,H,W
        if arr.shape[1] in (1, 3):
            return arr  # B,C,H,W
        # else maybe C,B,H,W
        if arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 0, 2, 3))  # B,C,H,W
            return arr
    raise ValueError(f"Cannot prepare array with shape {arr.shape}")


def _make_grid(bchw: np.ndarray) -> np.ndarray:
    """Make grid image from BxCxHxW array."""
    b, c, h, w = bchw.shape

    # Create a more compact grid layout
    # For small batch sizes, prefer horizontal layout, except for 4 images (2x2)
    if b == 1:
        grid_cols, grid_rows = 1, 1
    elif b == 2:
        grid_cols, grid_rows = 2, 1  # side by side
    elif b == 3:
        grid_cols, grid_rows = 3, 1  # all in a row
    elif b == 4:
        grid_cols, grid_rows = 2, 2  # 2x2 grid
    else:
        # For larger batches, use a more square-like layout
        grid_cols = math.ceil(math.sqrt(b))
        grid_rows = math.ceil(b / grid_cols)

    # canvas initialised to zeros (black background)
    canvas = np.zeros((h * grid_rows, w * grid_cols, c), dtype=bchw.dtype)
    for idx in range(b):
        row, col = divmod(idx, grid_cols)
        img = _to_hwc(bchw[idx])
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w, :] = img
    return canvas


def _convert_float_to_int(arr: np.ndarray) -> np.ndarray:
    """Convert float arrays with values in 0-255 range to uint8."""
    if arr.dtype.kind == "f":  # float type
        arr_min, arr_max = arr.min(), arr.max()
        # Only convert if values are clearly in 0-255 range, not 0-1 range
        # We check if max > 1.5 to distinguish from normalized 0-1 arrays
        if arr_min >= -0.5 and arr_max > 1.5 and arr_max <= 255.5:
            return np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return arr


def _prepare_for_display(arr: np.ndarray) -> np.ndarray:
    arr = _prep(arr)
    if arr.ndim == 4:
        arr = _make_grid(arr)
    arr = _convert_float_to_int(arr)
    return arr


def _create_figure(tensor: Any, **imshow_kwargs) -> plt.Figure:
    """Create a matplotlib figure from tensor."""
    arr = _to_numpy(tensor)
    arr = _prepare_for_display(arr)

    # Set figure size to match exact pixel dimensions
    h, w = arr.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

    if arr.ndim == 2 or arr.shape[2] == 1:
        ax.imshow(arr.squeeze(), cmap="gray", **imshow_kwargs)
    else:
        ax.imshow(arr, **imshow_kwargs)
    ax.axis("off")

    # Remove all padding to ensure exact pixel dimensions
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def plot(tensor: Any, **imshow_kwargs) -> plt.Figure:
    """
    Display *tensor* using Matplotlib.

    Parameters
    ----------
    tensor : torch.Tensor | np.ndarray
        Image tensor of shape (*, H, W) or (*, C, H, W).
    **imshow_kwargs
        Extra arguments forwarded to plt.imshow.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _create_figure(tensor, **imshow_kwargs)
    plt.show()


def save(path_or_tensor: Any, tensor: Any | None = None, **imshow_kwargs) -> str:
    """
    Save *tensor* to *path*. Two call styles are supported::

        save('img.png', tensor)
        save(tensor)  # auto tmp path

    Parameters
    ----------
    path_or_tensor :
        Destination path or tensor (if path omitted).
    tensor :
        Tensor to save, or None if tensor is first positional argument.

    Returns
    -------
    str
        Resolved file path.
    """
    if tensor is None:
        tensor, path = path_or_tensor, None
    else:
        path = path_or_tensor  # type: ignore[assignment]

    fig = _create_figure(tensor, **imshow_kwargs)

    if path is None:
        fd, path = tempfile.mkstemp(suffix=".png", prefix="vizy-")
        os.close(fd)
    fig.savefig(path, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(path)
    return path
