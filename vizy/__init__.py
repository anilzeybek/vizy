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
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from vizy import format_detection

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

if torch is not None and Image is not None:
    type TensorLike = torch.Tensor | np.ndarray | Image.Image
elif torch is not None:
    type TensorLike = torch.Tensor | np.ndarray
elif Image is not None:
    type TensorLike = np.ndarray | Image.Image
else:
    type TensorLike = np.ndarray


def _is_sequence_of_tensors(x: TensorLike | Sequence[TensorLike]) -> bool:
    """Check if x is a list/tuple of torch.Tensor, np.ndarray, or PIL.Image."""
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


def _pad_to_common_size(numpy_arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Pad numpy arrays to have the same height and width dimensions."""
    if len(numpy_arrays) == 0:
        return numpy_arrays

    hw_pairs = []
    for arr in numpy_arrays:
        if arr.ndim == 2:
            h, w = arr.shape
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):
                h, w = arr.shape[1], arr.shape[2]
            else:  # HWC format
                h, w = arr.shape[0], arr.shape[1]
        else:
            raise ValueError(f"Expected 2D or 3D arrays, got {arr.ndim}D")
        hw_pairs.append((h, w))

    max_h = max(h for h, _ in hw_pairs)
    max_w = max(w for _, w in hw_pairs)

    padded_arrays = []
    for arr, (h, w) in zip(numpy_arrays, hw_pairs):
        pad_h = max_h - h
        pad_w = max_w - w

        if arr.ndim == 2:
            padded_arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):  # CHW format
                padded_arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            else:  # HWC format
                padded_arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)

        padded_arrays.append(padded_arr)
    return padded_arrays


def _to_numpy(x: TensorLike | Sequence[TensorLike]) -> np.ndarray:
    if _is_sequence_of_tensors(x):
        numpy_arrays = []
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
            arr = arr.squeeze()
            if arr.ndim not in (2, 3):
                raise ValueError(
                    f"Each tensor in list must be 2D or 3D after squeezing, got {arr.ndim}D with shape {arr.shape}"
                )

            numpy_arrays.append(arr)

        numpy_arrays = _pad_to_common_size(numpy_arrays)
        stacked_numpy_array = np.stack(numpy_arrays, axis=0)  # Creates (B, ...) format
        return stacked_numpy_array

    # Handle single tensor/array/image
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif Image is not None and isinstance(x, Image.Image):
        x = np.array(x)

    if not isinstance(x, np.ndarray):
        raise TypeError("Expected torch.Tensor | np.ndarray | PIL.Image | sequence of these types")
    return x


def _normalize_array_format(numpy_arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Convert any given numpy array to HW/BHW/HWC/BHWC format and return whether it requires grid layout."""
    numpy_arr = numpy_arr.squeeze()

    if numpy_arr.ndim == 2:
        return numpy_arr, False

    if numpy_arr.ndim == 3:
        format_type = format_detection.detect_3d_array_format(numpy_arr)
        if format_type == format_detection.Array3DFormat.HWC:
            return numpy_arr, False
        if format_type == format_detection.Array3DFormat.CHW:
            return numpy_arr.transpose(1, 2, 0), False
        if format_type == format_detection.Array3DFormat.BHW:
            return numpy_arr, True
        if format_type == format_detection.Array3DFormat.HWB:
            return numpy_arr.transpose(2, 0, 1), True

    if numpy_arr.ndim == 4:
        format_type = format_detection.detect_4d_array_format(numpy_arr)
        if format_type == format_detection.Array4DFormat.HWCB:
            return numpy_arr.transpose(3, 0, 1, 2), True
        if format_type == format_detection.Array4DFormat.CHWB:
            return numpy_arr.transpose(3, 1, 2, 0), True
        if format_type == format_detection.Array4DFormat.BHWC:
            return numpy_arr, True
        if format_type == format_detection.Array4DFormat.BCHW:
            return numpy_arr.transpose(0, 2, 3, 1), True
        if format_type == format_detection.Array4DFormat.CBHW:
            return numpy_arr.transpose(1, 2, 3, 0), True

    raise ValueError(f"Cannot prepare array with shape {numpy_arr.shape}")


def _make_grid(numpy_arr: np.ndarray) -> np.ndarray:
    """Make grid image from BHWC/BHW array.

    Arranges multiple images in a grid layout with the following properties:
    - Single image remains unchanged
    - 2-3 images arranged horizontally in a row
    - 4 images arranged in a 2x2 grid
    - Larger batches arranged in a roughly square grid
    - Maintains original image dimensions and channels
    - Uses black background for empty grid positions
    """
    if numpy_arr.ndim == 4:
        b, h, w, c = numpy_arr.shape
    else:
        b, h, w = numpy_arr.shape
        c = 1

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
    canvas = np.zeros((h * grid_rows, w * grid_cols, c), dtype=numpy_arr.dtype)
    for idx in range(b):
        row, col = divmod(idx, grid_cols)
        img = numpy_arr[idx]
        if img.ndim == 2:
            img = img[..., np.newaxis]
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w, :] = img
    return canvas


def _convert_float_arr_to_int_arr(numpy_arr: np.ndarray) -> np.ndarray:
    """Convert float arrays with values in 0-255 range to uint8."""
    if numpy_arr.dtype.kind == "f":  # float type
        arr_min, arr_max = numpy_arr.min(), numpy_arr.max()
        # Only convert if values are clearly in 0-255 range, not 0-1 range
        # We check if max > 1.5 to distinguish from normalized 0-1 arrays
        if arr_min >= -0.5 and arr_max > 1.5 and arr_max <= 255.5:
            return np.clip(np.round(numpy_arr), 0, 255).astype(np.uint8)
    return numpy_arr


def _to_plottable_int_arr(numpy_arr: np.ndarray) -> np.ndarray:
    numpy_arr, requires_grid = _normalize_array_format(numpy_arr)
    if requires_grid:
        numpy_arr = _make_grid(numpy_arr)
    numpy_arr = _convert_float_arr_to_int_arr(numpy_arr)
    return numpy_arr


def _create_figure(tensor: TensorLike | Sequence[TensorLike], **imshow_kwargs) -> plt.Figure:
    """Create a matplotlib figure from tensor."""
    numpy_arr = _to_numpy(tensor)
    numpy_arr = _to_plottable_int_arr(numpy_arr)

    # Set figure size to match exact pixel dimensions
    h, w = numpy_arr.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

    if numpy_arr.ndim == 2 or numpy_arr.shape[2] == 1:
        ax.imshow(numpy_arr.squeeze(), cmap="gray", **imshow_kwargs)
    else:
        ax.imshow(numpy_arr, **imshow_kwargs)
    ax.axis("off")

    # Remove all padding to ensure exact pixel dimensions
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def plot(tensor: TensorLike | Sequence[TensorLike], **imshow_kwargs) -> plt.Figure:
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


def save(
    path_or_tensor: str | TensorLike | Sequence[TensorLike], tensor: TensorLike | None = None, **imshow_kwargs
) -> str:
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


def summary(tensor: TensorLike | Sequence[TensorLike]) -> None:
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
