"""vizy: One-line tensor visualization for PyTorch and NumPy.

Install
-------
pip install vizy   # distribution name
import vizy

API
---
vizy.plot(tensor)  # show tensor as image or grid
vizy.save(path_or_tensor, tensor=None)  # save to file

If *tensor* is 4-D we assume shape is either (B, C, H, W) or (C, B, H, W) with C in {1,3}.
For ndarray/tensors of 2-D or 3-D we transpose to (H, W, C) format.
Supports torch.Tensor, numpy.ndarray, PIL.Image inputs, and lists/sequences of these types.
"""

import math
import os
import tempfile
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from vizy import format_detection

try:
    import torch
except ModuleNotFoundError:
    torch = None


__all__: Sequence[str] = ("plot", "save", "summary")
__version__: str = "0.2.0"

if TYPE_CHECKING:
    import torch as _torch_mod

    type _TorchTensor = _torch_mod.Tensor
else:
    type _TorchTensor = np.ndarray
type TensorLike = _TorchTensor | Image.Image | NDArray[np.number]


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
        is_pil = isinstance(item, Image.Image)
        if not (is_tensor or is_array or is_pil):
            return False
    return True


def _pad_to_common_size(numpy_arrays: list[NDArray[np.number]]) -> list[NDArray[np.number]]:
    """Pad numpy arrays to have the same height and width dimensions."""
    if len(numpy_arrays) == 0:
        return numpy_arrays

    hw_pairs: list[tuple[int, int]] = []
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

    padded_arrays: list[NDArray[np.number]] = []
    for arr, (h, w) in zip(numpy_arrays, hw_pairs, strict=True):
        pad_h = max_h - h
        pad_w = max_w - w

        padded_arr: NDArray[np.number] | None = None
        if arr.ndim == 2:
            padded_arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):  # CHW format
                padded_arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            else:  # HWC format
                padded_arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)

        assert padded_arr is not None
        padded_arrays.append(padded_arr)
    return padded_arrays


def _to_numpy(x: TensorLike | Sequence[TensorLike]) -> NDArray[np.number]:
    if _is_sequence_of_tensors(x):
        assert isinstance(x, Sequence)
        numpy_arrays: list[NDArray[np.number]] = []
        for item in x:
            if torch is not None and isinstance(item, torch.Tensor):
                arr = item.detach().cpu().numpy()
            elif isinstance(item, Image.Image):
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
        return np.stack(numpy_arrays, axis=0)  # Creates (B, ...) format

    # Handle single tensor/array/image
    if torch is not None and isinstance(x, torch.Tensor):
        np_arr = x.detach().cpu().numpy()
    elif isinstance(x, Image.Image):
        np_arr = np.array(x)
    elif isinstance(x, np.ndarray):
        np_arr = x
    else:
        raise TypeError(
            f"Unsupported type: {type(x)}, expected torch.Tensor | np.ndarray | PIL.Image | sequence of these types"
        )

    return np_arr


def _normalize_3d_array(numpy_arr: NDArray[np.number]) -> tuple[NDArray[np.number], bool]:
    """Normalize a 3D array to HWC or BHW format and return whether it requires grid layout."""
    format_type = format_detection.detect_3d_array_format(numpy_arr)
    match format_type:
        case format_detection.Array3DFormat.HWC:
            return numpy_arr, False
        case format_detection.Array3DFormat.CHW:
            return numpy_arr.transpose(1, 2, 0), False
        case format_detection.Array3DFormat.BHW:
            return numpy_arr, True
        case format_detection.Array3DFormat.HWB:
            return numpy_arr.transpose(2, 0, 1), True


def _normalize_4d_array(numpy_arr: NDArray[np.number]) -> tuple[NDArray[np.number], bool]:
    """Normalize a 4D array to BHWC format and return whether it requires grid layout."""
    format_type = format_detection.detect_4d_array_format(numpy_arr)
    match format_type:
        case format_detection.Array4DFormat.HWCB:
            return numpy_arr.transpose(3, 0, 1, 2), True
        case format_detection.Array4DFormat.CHWB:
            return numpy_arr.transpose(3, 1, 2, 0), True
        case format_detection.Array4DFormat.BHWC:
            return numpy_arr, True
        case format_detection.Array4DFormat.BCHW:
            return numpy_arr.transpose(0, 2, 3, 1), True
        case format_detection.Array4DFormat.CBHW:
            return numpy_arr.transpose(1, 2, 3, 0), True


def _normalize_array_format(numpy_arr: NDArray[np.number]) -> tuple[NDArray[np.number], bool]:
    """Convert any given numpy array to HW/BHW/HWC/BHWC format and return whether it requires grid layout."""
    numpy_arr = numpy_arr.squeeze()

    if numpy_arr.ndim == 2:
        return numpy_arr, False
    if numpy_arr.ndim == 3:
        return _normalize_3d_array(numpy_arr)
    if numpy_arr.ndim == 4:
        return _normalize_4d_array(numpy_arr)
    raise ValueError(f"Cannot prepare array with {numpy_arr.ndim} dimensions")


def _make_grid(numpy_arr: NDArray[np.number]) -> NDArray[np.number]:
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


def _convert_float_to_uint8(numpy_arr: NDArray[np.number]) -> NDArray[np.uint8]:
    """Convert float array to uint8, handling various ranges."""
    arr_min = numpy_arr.min()
    arr_max = numpy_arr.max()
    # Check if values are in 0-255 range (not normalized 0-1)
    # We check if max > 1.5 to distinguish from normalized 0-1 arrays
    # Require arr_min >= 0 to ensure no negative values (which would indicate
    # the array is not in 0-255 pixel range and should be normalized instead)
    if arr_min >= 0 and arr_max > 1.5 and arr_max <= 255.5:
        # Already in 0-255 range, convert directly
        return np.clip(np.round(numpy_arr), 0, 255).astype(np.uint8)

    # Likely normalized 0-1 range or other range, normalize to 0-255
    # if all values between 0 and 1 (inclusive):
    if np.all(numpy_arr >= 0) and np.all(numpy_arr <= 1):
        return np.clip(np.round(numpy_arr * 255), 0, 255).astype(np.uint8)
    if arr_max > arr_min:  # Avoid division by zero
        normalized = (numpy_arr - arr_min) / (arr_max - arr_min)
        return np.clip(np.round(normalized * 255), 0, 255).astype(np.uint8)
    # All values are the same
    # If value is exactly 1.0, treat as normalized (scale to 255)
    # Otherwise, clip to 0-255 range
    if arr_min == 1.0:
        return np.clip(np.round(numpy_arr * 255), 0, 255).astype(np.uint8)
    return np.clip(np.round(numpy_arr), 0, 255).astype(np.uint8)


def _force_np_arr_to_int_arr(numpy_arr: NDArray[np.number]) -> NDArray[np.uint8]:
    """Force numpy array to uint8."""
    if numpy_arr.dtype == np.uint8:
        if np.max(numpy_arr) <= 1:
            numpy_arr *= 255
        return cast(NDArray[np.uint8], numpy_arr)
    if numpy_arr.dtype.kind == "f":  # float type
        return _convert_float_to_uint8(numpy_arr)
    if numpy_arr.dtype.kind in ("i", "u"):  # signed or unsigned integer
        # Convert other integer types to uint8 with clipping
        if np.max(numpy_arr) <= 1:
            numpy_arr *= 255
        return np.clip(numpy_arr, 0, 255).astype(np.uint8)
    if numpy_arr.dtype.kind == "b":  # boolean
        return numpy_arr.astype(np.uint8) * 255

    raise ValueError(f"Unsupported dtype for conversion to uint8: {numpy_arr.dtype}")


def _to_plottable_int_arr(numpy_arr: NDArray[np.number]) -> NDArray[np.uint8]:
    numpy_arr, requires_grid = _normalize_array_format(numpy_arr)
    if requires_grid:
        numpy_arr = _make_grid(numpy_arr)
    return _force_np_arr_to_int_arr(numpy_arr)


def _numpy_to_pil_image(numpy_arr: NDArray[np.uint8]) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if numpy_arr.ndim == 2:
        # Grayscale image
        return Image.fromarray(numpy_arr, mode="L")

    if numpy_arr.ndim == 3:
        if numpy_arr.shape[2] == 1:
            # Single channel, convert to 2D
            return Image.fromarray(numpy_arr.squeeze(), mode="L")
        if numpy_arr.shape[2] == 3:
            # RGB image
            return Image.fromarray(numpy_arr, mode="RGB")
        if numpy_arr.shape[2] == 4:
            # RGBA image
            return Image.fromarray(numpy_arr, mode="RGBA")
        raise ValueError(f"Unsupported number of channels: {numpy_arr.shape[2]}")

    raise ValueError(f"Unsupported array dimensions: {numpy_arr.ndim}")


def _tensor_to_pil_image(tensor: TensorLike | Sequence[TensorLike]) -> Image.Image:
    numpy_arr = _to_numpy(tensor)
    if np.any(np.isnan(numpy_arr)):
        raise ValueError("Cannot plot array with NaN values")
    plottable_numpy_arr = _to_plottable_int_arr(numpy_arr)
    return _numpy_to_pil_image(plottable_numpy_arr)


def plot(tensor: TensorLike | Sequence[TensorLike]) -> None:
    """Display *tensor* using PIL/Pillow (opens system image viewer).

    In Jupyter notebooks, displays inline. Otherwise, opens system image viewer.

    Parameters
    ----------
    tensor : torch.Tensor | np.ndarray | PIL.Image | sequence of these
        Image tensor of shape (*, H, W) or (*, C, H, W), PIL Image, or a
        list/tuple of 2D/3D tensors. For lists with mismatched dimensions,
        images will be padded to the largest size.

    """
    pil_image = _tensor_to_pil_image(tensor)

    # Check if we're in a Jupyter notebook environment
    try:
        import sys

        from IPython.display import display

        if "ipykernel" in sys.modules:
            display(pil_image)
            return
    except ImportError:
        pass

    # Fall back to system image viewer
    pil_image.show()


def save(
    path_or_tensor: str | TensorLike | Sequence[TensorLike],
    tensor: TensorLike | Sequence[TensorLike] | None = None,
) -> str:
    """Save *tensor* to *path*.

    Two call styles are supported::

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
        assert not isinstance(path_or_tensor, str)
        tensor, path = path_or_tensor, None
    else:
        assert isinstance(path_or_tensor, str)
        path = path_or_tensor

    if path is None:
        fd, path = tempfile.mkstemp(suffix=".png", prefix="vizy-")
        os.close(fd)

    pil_image = _tensor_to_pil_image(tensor)
    pil_image.save(path)

    print(path)
    return path


def _get_tensor_info(item: TensorLike) -> tuple[str, str, NDArray[np.number], str] | None:
    """Extract type, device_info, numpy array, and dtype string from a tensor."""
    if torch is not None and isinstance(item, torch.Tensor):
        item_type = "torch.Tensor"
        device_info = f" (device: {item.device})" if hasattr(item, "device") else ""
        arr = item.detach().cpu().numpy()
        dtype_str = str(item.dtype)
        return item_type, device_info, arr, dtype_str
    if isinstance(item, Image.Image):
        item_type = "PIL.Image"
        device_info = f" (mode: {item.mode})"
        arr = np.array(item)
        dtype_str = str(arr.dtype)
        return item_type, device_info, arr, dtype_str
    if isinstance(item, np.ndarray):
        item_type = "numpy.ndarray"
        device_info = ""
        arr = item
        dtype_str = str(item.dtype)
        return item_type, device_info, arr, dtype_str
    return None


def _print_array_stats(arr: NDArray[np.number], *, include_unique: bool = False) -> None:
    """Print statistics about an array (shape, dtype, range, optionally unique values)."""
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    if arr.size > 0:
        arr_min = arr.min()
        arr_max = arr.max()
        print(f"Range: {arr_min} - {arr_max}")
        if include_unique and arr.dtype.kind in ("i", "u"):
            unique_count = len(np.unique(arr))
            print(f"Number of unique values: {unique_count}")
    else:
        print("Range: N/A (empty array)")


def _summary_sequence(tensor: Sequence[TensorLike]) -> None:
    """Print summary for a sequence of tensors."""
    print(f"Type: Sequence ({type(tensor).__name__}) of {len(tensor)} tensors")
    print("Individual tensor info:")
    for i, item in enumerate(tensor):
        print(f"  [{i}]:", end=" ")
        info = _get_tensor_info(item)
        if info is None:
            print(f"Unsupported type: {type(item)}")
            continue
        item_type, device_info, arr, dtype_str = info
        print(f"{item_type}{device_info}, Shape: {arr.shape}, Dtype: {dtype_str}")

    batch_arr = _to_numpy(tensor)
    print("\nProcessed as batch:")
    _print_array_stats(batch_arr)


def _summary_single(tensor: TensorLike) -> None:
    """Print summary for a single tensor."""
    info = _get_tensor_info(tensor)
    if info is None:
        raise TypeError("Expected torch.Tensor | np.ndarray | PIL.Image | sequence of these")
    array_type, device_info, arr, dtype_str = info

    print(f"Type: {array_type}{device_info}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {dtype_str}")
    if arr.size > 0:
        arr_min = arr.min()
        arr_max = arr.max()
        print(f"Range: {arr_min} - {arr_max}")
        if arr.dtype.kind in ("i", "u"):
            unique_count = len(np.unique(arr))
            print(f"Number of unique values: {unique_count}")
    else:
        print("Range: N/A (empty array)")


def summary(tensor: TensorLike | Sequence[TensorLike]) -> None:
    """Print summary information about a tensor or array.

    Parameters
    ----------
    tensor : torch.Tensor | np.ndarray | PIL.Image | sequence of these
        Tensor, array, PIL Image, or list/tuple of these to summarize.

    """
    if _is_sequence_of_tensors(tensor):
        assert isinstance(tensor, Sequence)
        _summary_sequence(tensor)
    else:
        _summary_single(cast(TensorLike, tensor))
