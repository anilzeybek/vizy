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
Supports torch.Tensor, numpy.ndarray, PIL.Image inputs, and lists/sequences of these types.
"""

import math
import os
import tempfile
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .format_detection import smart_3d_format_detection, smart_4d_format_detection

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Image = None  # type: ignore

__all__: Sequence[str] = ("plot", "save", "summary")
__version__: str = "0.2.0"


def _is_sequence_of_tensors(x: Any) -> bool:
    """Check if x is a list/tuple of tensors, arrays, or PIL Images."""
    if not isinstance(x, (list, tuple)):
        return False
    if len(x) == 0:
        return False

    # Check if all elements are valid tensor types
    for item in x:
        is_tensor = torch is not None and isinstance(item, torch.Tensor)
        is_array = isinstance(item, np.ndarray)
        is_pil = Image is not None and isinstance(item, Image.Image)
        if not (is_tensor or is_array or is_pil):
            return False
    return True


def _pad_to_common_size(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Pad arrays to have the same height and width (last two dimensions)."""
    if len(arrays) == 0:
        return arrays

    # Find maximum dimensions
    max_h = max(arr.shape[-2] for arr in arrays)
    max_w = max(arr.shape[-1] for arr in arrays)

    padded_arrays = []
    for arr in arrays:
        if arr.ndim == 2:
            h, w = arr.shape
            pad_h = max_h - h
            pad_w = max_w - w
            # Pad with zeros (black) on bottom and right
            padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):  # CHW format
                c, h, w = arr.shape
                pad_h = max_h - h
                pad_w = max_w - w
                padded = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            else:  # HWC format
                h, w, c = arr.shape
                pad_h = max_h - h
                pad_w = max_w - w
                padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        else:
            raise ValueError(f"Expected 2D or 3D arrays, got {arr.ndim}D")

        padded_arrays.append(padded)
    return padded_arrays


def _to_numpy(x: Any) -> np.ndarray:
    """Convert x to NumPy array, detaching from torch if needed. Handles lists/sequences of tensors."""
    # Handle lists/sequences of tensors
    if _is_sequence_of_tensors(x):
        # Convert each item to numpy and validate dimensions
        arrays = []
        for item in x:
            if torch is not None and isinstance(item, torch.Tensor):
                arr = item.detach().cpu().numpy()
            elif Image is not None and isinstance(item, Image.Image):
                arr = np.array(item)
            elif isinstance(item, np.ndarray):
                arr = item
            else:
                raise TypeError(f"Unsupported type in sequence: {type(item)}")

            # Validate that each tensor is 2D or 3D (no batches in the list)
            arr = arr.squeeze()  # Remove singleton dimensions
            if arr.ndim not in (2, 3):
                raise ValueError(
                    f"Each tensor in list must be 2D or 3D after squeezing, got {arr.ndim}D with shape {arr.shape}"
                )

            arrays.append(arr)

        # Pad arrays to common size
        arrays = _pad_to_common_size(arrays)

        # Stack arrays to create a batch dimension
        # All arrays should now have the same shape
        stacked = np.stack(arrays, axis=0)  # Creates (B, ...) format
        return stacked

    # Handle single tensor/array/image
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif Image is not None and isinstance(x, Image.Image):
        # Convert PIL Image to numpy array
        x = np.array(x)
    if not isinstance(x, np.ndarray):
        raise TypeError("Expected torch.Tensor | np.ndarray | PIL.Image | sequence of these types")
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
    """Prepare array for visualization by:
    - Squeezing singleton dimensions
    - Converting 2D/3D arrays to HWC format using _to_hwc
    - Ensuring 4D arrays are in BHWC format
    - Raising error for unsupported shapes
    """
    arr = arr.squeeze()
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Special handling for ambiguous (3, H, W) case
        if arr.shape[0] == 3:
            format_type = smart_3d_format_detection(arr)
            if format_type == "rgb":
                # Treat as single RGB image: (3, H, W) -> (H, W, 3)
                return np.transpose(arr, (1, 2, 0))
            else:
                # Treat as batch: (3, H, W) -> (3, H, W, 1) -> continue to 4D handling
                return arr[:, :, :, np.newaxis]  # Add channel dimension
        else:
            # Non-ambiguous 3D case
            return _to_hwc(arr)

    if arr.ndim == 4:
        # Handle the ambiguous (3, 3, H, W) case
        if arr.shape[0] == 3 and arr.shape[1] == 3:
            format_type = smart_4d_format_detection(arr)
            if format_type == "CBHW":
                # Convert C,B,H,W -> B,H,W,C
                arr = np.transpose(arr, (1, 2, 3, 0))
            else:
                # Convert B,C,H,W -> B,H,W,C
                arr = np.transpose(arr, (0, 2, 3, 1))
            return arr

        # Non-ambiguous 4D cases
        # try B,C,H,W
        if arr.shape[1] in (1, 3):
            # Convert B,C,H,W -> B,H,W,C
            arr = np.transpose(arr, (0, 2, 3, 1))
            return arr
        # else maybe C,B,H,W
        if arr.shape[0] in (1, 3):
            # Convert C,B,H,W -> B,H,W,C
            arr = np.transpose(arr, (1, 2, 3, 0))
            return arr

    raise ValueError(f"Cannot prepare array with shape {arr.shape}")


def _make_grid(bhwc: np.ndarray) -> np.ndarray:
    """Make grid image from BxHxWxC array.

    Arranges multiple images in a grid layout with the following properties:
    - Single image remains unchanged
    - 2-3 images arranged horizontally in a row
    - 4 images arranged in a 2x2 grid
    - Larger batches arranged in a roughly square grid
    - Maintains original image dimensions and channels
    - Uses black background for empty grid positions
    """
    b, h, w, c = bhwc.shape

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
    canvas = np.zeros((h * grid_rows, w * grid_cols, c), dtype=bhwc.dtype)
    for idx in range(b):
        row, col = divmod(idx, grid_cols)
        img = bhwc[idx]  # Already in HWC format
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
    tensor : torch.Tensor | np.ndarray | PIL.Image | sequence of these
        Image tensor of shape (*, H, W) or (*, C, H, W), PIL Image, or a
        list/tuple of 2D/3D tensors. For lists with mismatched dimensions,
        images will be padded to the largest size.
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
    tensor : torch.Tensor | np.ndarray | PIL.Image | sequence of these | None
        Tensor to save, or None if tensor is first positional argument.
        For lists with mismatched dimensions, images will be padded to the largest size.

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


def summary(tensor: Any) -> None:
    """
    Print summary information about a tensor or array.

    Parameters
    ----------
    tensor : torch.Tensor | np.ndarray | PIL.Image | sequence of these
        Tensor, array, PIL Image, or list/tuple of these to summarize.
    """
    if _is_sequence_of_tensors(tensor):
        print(f"Type: Sequence ({type(tensor).__name__}) of {len(tensor)} tensors")
        print("Individual tensor info:")
        for i, item in enumerate(tensor):
            print(f"  [{i}]:", end=" ")
            # Get basic info for each item
            if torch is not None and isinstance(item, torch.Tensor):
                item_type = "torch.Tensor"
                device_info = f" (device: {item.device})" if hasattr(item, "device") else ""
                arr = item.detach().cpu().numpy()
                dtype_str = str(item.dtype)
            elif Image is not None and isinstance(item, Image.Image):
                item_type = "PIL.Image"
                device_info = f" (mode: {item.mode})"
                arr = np.array(item)
                dtype_str = str(arr.dtype)
            elif isinstance(item, np.ndarray):
                item_type = "numpy.ndarray"
                device_info = ""
                arr = item
                dtype_str = str(item.dtype)
            else:
                print(f"Unsupported type: {type(item)}")
                continue

            print(f"{item_type}{device_info}, Shape: {arr.shape}, Dtype: {dtype_str}")

        # Also show the stacked/processed version
        print("\nProcessed as batch:")
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        if arr.size > 0:
            arr_min, arr_max = arr.min(), arr.max()
            print(f"Range: {arr_min} - {arr_max}")
        return

    # Determine the original type for single tensors
    if torch is not None and isinstance(tensor, torch.Tensor):
        array_type = "torch.Tensor"
        device_info = f" (device: {tensor.device})" if hasattr(tensor, "device") else ""
        # Convert to numpy for analysis but keep original for type info
        arr = tensor.detach().cpu().numpy()
        dtype_str = str(tensor.dtype)
    elif Image is not None and isinstance(tensor, Image.Image):
        array_type = "PIL.Image"
        device_info = f" (mode: {tensor.mode})"
        # Convert to numpy for analysis
        arr = np.array(tensor)
        dtype_str = str(arr.dtype)
    elif isinstance(tensor, np.ndarray):
        array_type = "numpy.ndarray"
        device_info = ""
        arr = tensor
        dtype_str = str(tensor.dtype)
    else:
        raise TypeError("Expected torch.Tensor | np.ndarray | PIL.Image | sequence of these")

    # Basic info
    print(f"Type: {array_type}{device_info}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {dtype_str}")

    # Range (min - max)
    if arr.size > 0:  # Only if array is not empty
        arr_min, arr_max = arr.min(), arr.max()
        print(f"Range: {arr_min} - {arr_max}")

        # Number of unique values for integer dtypes
        if arr.dtype.kind in ("i", "u"):  # signed or unsigned integer
            unique_count = len(np.unique(arr))
            print(f"Number of unique values: {unique_count}")
    else:
        print("Range: N/A (empty array)")
