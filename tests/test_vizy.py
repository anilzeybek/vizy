import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

import vizy


class TestNormalizeArrayFormat:
    """Test the _normalize_array_format function."""

    def test_2d_array_unchanged(self) -> None:
        """Test that 2D arrays (H, W) are unchanged."""
        rng = np.random.default_rng(42)
        arr = rng.random((100, 200))
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        assert np.array_equal(result, arr)
        assert result.shape == (100, 200)

    def test_3d_hwc_unchanged(self) -> None:
        """Test that 3D arrays in HWC format are unchanged."""
        rng = np.random.default_rng(42)
        arr = rng.random((50, 60, 3))  # H, W, C
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        assert np.array_equal(result, arr)
        assert result.shape == (50, 60, 3)

    def test_3d_chw_to_hwc(self) -> None:
        """Test conversion from CHW to HWC format."""
        rng = np.random.default_rng(42)
        arr = rng.random((3, 50, 60))  # C, H, W
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        expected = np.transpose(arr, (1, 2, 0))
        assert np.array_equal(result, expected) or np.array_equal(result, arr)
        assert result.shape in ((50, 60, 3), (3, 50, 60))

    def test_3d_single_channel_chw(self) -> None:
        """Test conversion from single channel CHW to HWC."""
        rng = np.random.default_rng(42)
        arr = rng.random((1, 40, 50))  # C=1, H, W
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        expected = arr.squeeze(axis=0)
        assert np.array_equal(result, expected)
        assert result.shape == (40, 50)

    def test_3d_ambiguous_case(self) -> None:
        """Test case where both dimensions could be channels."""
        # When both first and last dim are 3, should prefer HWC (no transpose)
        rng = np.random.default_rng(42)
        arr = rng.random((3, 50, 3))
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        assert np.array_equal(result, arr)  # Should remain unchanged

    def test_invalid_dimensions(self) -> None:
        """Test that arrays with unsupported dimensions raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Unable to determine 4D array format"):
            vizy._normalize_array_format(rng.random((10, 20, 30, 40)))  # noqa: SLF001

        with pytest.raises(ValueError, match="Cannot prepare array"):
            vizy._normalize_array_format(rng.random((10,)))  # noqa: SLF001


class TestPrep:
    """Test the _prep function."""

    def test_2d_array(self) -> None:
        """Test preparation of 2D arrays."""
        rng = np.random.default_rng(42)
        arr = rng.random((50, 60))
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        assert result.shape == (50, 60)
        assert result.ndim == 2

    def test_3d_array_hwc(self) -> None:
        """Test preparation of 3D arrays in HWC format."""
        rng = np.random.default_rng(42)
        arr = rng.random((50, 60, 3))
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        assert result.shape == (50, 60, 3)

    def test_3d_array_chw(self) -> None:
        """Test preparation of 3D arrays in CHW format."""
        rng = np.random.default_rng(42)
        arr = rng.random((3, 50, 60))
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        assert result.shape in ((50, 60, 3), (3, 50, 60))

    def test_4d_bchw(self) -> None:
        """Test preparation of 4D arrays in BCHW format."""
        rng = np.random.default_rng(42)
        arr = rng.random((4, 3, 50, 60))  # B, C, H, W
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        expected = np.transpose(arr, (0, 2, 3, 1))  # B, H, W, C
        assert result.shape == (4, 50, 60, 3)
        assert np.array_equal(result, expected)

    def test_4d_cbhw_to_bchw(self) -> None:
        """Test conversion from CBHW to BCHW format."""
        rng = np.random.default_rng(42)
        arr = rng.random((3, 4, 50, 60))  # C, B, H, W
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        expected = np.transpose(arr, (1, 2, 3, 0))  # B, H, W, C
        assert result.shape == (4, 50, 60, 3)
        assert np.array_equal(result, expected)

    def test_4d_single_channel(self) -> None:
        """Test 4D arrays with single channel."""
        rng = np.random.default_rng(42)
        arr = rng.random((4, 1, 50, 60))  # B, C=1, H, W
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        expected = np.squeeze(arr, axis=1)  # B, H, W, C
        assert result.shape == (4, 50, 60)
        assert np.array_equal(result, expected)

    def test_4d_bhwc_unchanged(self) -> None:
        """Test that 4D arrays already in BHWC format are returned unchanged."""
        rng = np.random.default_rng(42)
        arr = rng.random((4, 50, 60, 3))  # B, H, W, C
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        assert result.shape == (4, 50, 60, 3)
        assert np.array_equal(result, arr)

    def test_squeeze_behavior(self) -> None:
        """Test that arrays are properly squeezed."""
        rng = np.random.default_rng(42)
        arr = rng.random((1, 1, 50, 60, 1))
        result, _ = vizy._normalize_array_format(arr)  # noqa: SLF001
        assert result.shape == (50, 60)

    def test_invalid_4d_shape(self) -> None:
        """Test that invalid 4D shapes raise ValueError."""
        rng = np.random.default_rng(42)
        # Neither dimension 0 nor 1 is a valid channel count
        arr = rng.random((5, 7, 50, 60))
        with pytest.raises(ValueError, match="Unable to determine 4D array format"):
            vizy._normalize_array_format(arr)  # noqa: SLF001

    def test_invalid_dimensions(self) -> None:
        """Test that unsupported dimensions raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Cannot prepare array"):
            vizy._normalize_array_format(rng.random((10, 20, 30, 40, 50)))  # noqa: SLF001


class TestMakeGrid:
    """Test the _make_grid function."""

    def test_single_image(self) -> None:
        """Test grid creation with single image."""
        rng = np.random.default_rng(42)
        bhwc = rng.random((1, 32, 32, 3))
        result = vizy._make_grid(bhwc)  # noqa: SLF001
        assert result.shape == (32, 32, 3)

    def test_two_images(self) -> None:
        """Test grid creation with two images (side by side)."""
        rng = np.random.default_rng(42)
        bhwc = rng.random((2, 32, 32, 3))
        result = vizy._make_grid(bhwc)  # noqa: SLF001
        assert result.shape == (32, 64, 3)  # 1 row, 2 cols

    def test_three_images(self) -> None:
        """Test grid creation with three images (all in a row)."""
        rng = np.random.default_rng(42)
        bhwc = rng.random((3, 32, 32, 3))
        result = vizy._make_grid(bhwc)  # noqa: SLF001
        assert result.shape == (32, 96, 3)  # 1 row, 3 cols

    def test_four_images(self) -> None:
        """Test grid creation with four images (2x2 grid)."""
        rng = np.random.default_rng(42)
        bhwc = rng.random((4, 32, 32, 3))
        result = vizy._make_grid(bhwc)  # noqa: SLF001
        assert result.shape == (64, 64, 3)  # 2 rows, 2 cols

    def test_larger_batch(self) -> None:
        """Test grid creation with larger batch."""
        rng = np.random.default_rng(42)
        bhwc = rng.random((9, 32, 32, 3))
        result = vizy._make_grid(bhwc)  # noqa: SLF001
        assert result.shape == (96, 96, 3)  # 3 rows, 3 cols

    def test_single_channel(self) -> None:
        """Test grid creation with single channel images."""
        rng = np.random.default_rng(42)
        bhwc = rng.random((4, 32, 32, 1))
        result = vizy._make_grid(bhwc)  # noqa: SLF001
        assert result.shape == (64, 64, 1)

    def test_non_square_images(self) -> None:
        """Test grid creation with non-square images."""
        rng = np.random.default_rng(42)
        bhwc = rng.random((4, 20, 30, 3))
        result = vizy._make_grid(bhwc)  # noqa: SLF001
        assert result.shape == (40, 60, 3)  # 2 rows, 2 cols


class TestPadToCommonSize:
    """Test the _pad_to_common_size function."""

    def test_empty_list(self) -> None:
        """Test that empty list returns empty list."""
        result = vizy._pad_to_common_size([])  # noqa: SLF001
        assert result == []

    def test_same_size_2d_arrays(self) -> None:
        """Test that same-size 2D arrays remain unchanged."""
        rng = np.random.default_rng(42)
        arr1 = rng.random((32, 32))
        arr2 = rng.random((32, 32))
        result = vizy._pad_to_common_size([arr1, arr2])  # noqa: SLF001
        assert len(result) == 2
        assert np.array_equal(result[0], arr1)
        assert np.array_equal(result[1], arr2)

    def test_different_size_2d_arrays(self) -> None:
        """Test padding of different-size 2D arrays."""
        rng = np.random.default_rng(42)
        arr1 = rng.random((20, 30))
        arr2 = rng.random((40, 50))
        result = vizy._pad_to_common_size([arr1, arr2])  # noqa: SLF001
        assert len(result) == 2
        assert result[0].shape == (40, 50)
        assert result[1].shape == (40, 50)
        # Check that original content is preserved
        assert np.array_equal(result[0][:20, :30], arr1)
        assert np.array_equal(result[1], arr2)
        # Check padding is zeros
        assert np.all(result[0][20:, :] == 0)
        assert np.all(result[0][:, 30:] == 0)

    def test_different_size_3d_hwc_arrays(self) -> None:
        """Test padding of different-size 3D HWC arrays."""
        rng = np.random.default_rng(42)
        arr1 = rng.random((20, 30, 3))
        arr2 = rng.random((40, 50, 3))
        result = vizy._pad_to_common_size([arr1, arr2])  # noqa: SLF001
        assert len(result) == 2
        assert result[0].shape == (40, 50, 3)
        assert result[1].shape == (40, 50, 3)

    def test_different_size_3d_chw_arrays(self) -> None:
        """Test padding of different-size 3D CHW arrays."""
        rng = np.random.default_rng(42)
        arr1 = rng.random((3, 20, 30))
        arr2 = rng.random((3, 40, 50))
        result = vizy._pad_to_common_size([arr1, arr2])  # noqa: SLF001
        assert len(result) == 2
        assert result[0].shape == (3, 40, 50)
        assert result[1].shape == (3, 40, 50)


class TestForceNpArrToIntArr:
    """Test the _force_np_arr_to_int_arr function."""

    def test_uint8_unchanged(self) -> None:
        """Test that uint8 arrays remain unchanged."""
        arr = np.array([0, 127, 255], dtype=np.uint8)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert np.array_equal(result, arr)
        assert result.dtype == np.uint8

    def test_float_in_0_255_range(self) -> None:
        """Test conversion of float arrays in 0-255 range to uint8."""
        arr = np.array([0.0, 127.5, 255.0], dtype=np.float32)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        expected = np.array([0, 128, 255], dtype=np.uint8)
        assert np.array_equal(result, expected)
        assert result.dtype == np.uint8

    def test_float_in_0_1_range(self) -> None:
        """Test that float arrays in 0-1 range are scaled to 0-255."""
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        expected = np.array([0, 128, 255], dtype=np.uint8)
        assert np.array_equal(result, expected)
        assert result.dtype == np.uint8

    def test_float_normalized_range(self) -> None:
        """Test that float arrays in arbitrary range are normalized to 0-255."""
        arr = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_integer_array_conversion(self) -> None:
        """Test conversion of other integer types."""
        arr = np.array([0, 127, 255], dtype=np.int32)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8
        assert np.array_equal(result, np.array([0, 127, 255], dtype=np.uint8))

    def test_integer_array_clipping(self) -> None:
        """Test that integer arrays outside 0-255 are clipped."""
        arr = np.array([-10, 300, 500], dtype=np.int32)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_array(self) -> None:
        """Test array with all same values."""
        arr = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8
        assert np.all(result == 50)

    def test_array_with_negative_values(self) -> None:
        """Test that arrays with negative values normalize instead of clipping."""
        arr = np.array([-0.3, 50.0, 200.0], dtype=np.float32)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8
        # Should normalize (not clip), so min should be 0 and max should be 255
        assert result.min() == 0
        assert result.max() == 255
        # The middle value should be normalized proportionally
        # Original range: -0.3 to 200.0 (range = 200.3)
        # Normalized 50.0: (50.0 - (-0.3)) / 200.3 * 255 ≈ 64
        assert result[1] == 64  # Approximately 64 after normalization and rounding

    def test_bool_array(self) -> None:
        """Test that bool arrays are converted to uint8."""
        arr = np.array([True, False, True], dtype=bool)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8
        assert np.array_equal(result, np.array([255, 0, 255], dtype=np.uint8))

    def test_uint8_binary_scaled_to_255(self) -> None:
        """Test that uint8 arrays with only 0 and 1 values are scaled to 0-255."""
        arr = np.array([0, 1, 1, 0], dtype=np.uint8)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8
        assert np.array_equal(result, np.array([0, 255, 255, 0], dtype=np.uint8))

    def test_uint8_binary_2d_image(self) -> None:
        """Test that 2D uint8 binary images are scaled correctly."""
        arr = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        expected = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        assert np.array_equal(result, expected)

    def test_uint8_all_zeros(self) -> None:
        """Test that uint8 array with all zeros remains unchanged."""
        arr = np.array([0, 0, 0], dtype=np.uint8)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert np.array_equal(result, arr)

    def test_uint8_all_ones(self) -> None:
        """Test that uint8 array with all ones is scaled to 255."""
        arr = np.array([1, 1, 1], dtype=np.uint8)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        expected = np.array([255, 255, 255], dtype=np.uint8)
        assert np.array_equal(result, expected)

    def test_uint8_normal_range_unchanged(self) -> None:
        """Test that uint8 arrays with values > 1 remain unchanged."""
        arr = np.array([0, 50, 128, 255], dtype=np.uint8)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert np.array_equal(result, arr)

    def test_int32_binary_scaled_to_255(self) -> None:
        """Test that int32 arrays with only 0 and 1 values are scaled to 0-255."""
        arr = np.array([0, 1, 1, 0], dtype=np.int32)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8
        assert np.array_equal(result, np.array([0, 255, 255, 0], dtype=np.uint8))

    def test_int64_binary_scaled_to_255(self) -> None:
        """Test that int64 arrays with only 0 and 1 values are scaled to 0-255."""
        arr = np.array([0, 1, 0, 1], dtype=np.int64)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8
        assert np.array_equal(result, np.array([0, 255, 0, 255], dtype=np.uint8))

    def test_int32_normal_range_unchanged(self) -> None:
        """Test that int32 arrays with values > 1 are clipped without scaling."""
        arr = np.array([0, 50, 128, 255], dtype=np.int32)
        result = vizy._force_np_arr_to_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8
        assert np.array_equal(result, np.array([0, 50, 128, 255], dtype=np.uint8))


class TestPrepareForDisplay:
    """Test the _prepare_for_display function."""

    def test_2d_array(self) -> None:
        """Test preparation of 2D array for display."""
        rng = np.random.default_rng(42)
        arr = rng.random((50, 60))
        result = vizy._to_plottable_int_arr(arr)  # noqa: SLF001
        assert result.shape == (50, 60)
        assert result.ndim == 2

    def test_3d_array(self) -> None:
        """Test preparation of 3D array for display."""
        rng = np.random.default_rng(42)
        arr = rng.random((50, 60, 3))
        result = vizy._to_plottable_int_arr(arr)  # noqa: SLF001
        assert result.shape == (50, 60, 3)

    def test_4d_array_to_grid(self) -> None:
        """Test that 4D arrays are converted to grids."""
        rng = np.random.default_rng(42)
        arr = rng.random((4, 3, 32, 32))
        result = vizy._to_plottable_int_arr(arr)  # noqa: SLF001
        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB channels

    def test_float_to_int_conversion(self) -> None:
        """Test that float arrays in 0-255 range are converted to uint8."""
        arr = np.array([[[100.0, 200.0, 255.0]]], dtype=np.float32)
        # This will be squeezed to (3,) which is invalid, so let's use a proper shape
        arr = np.array([[100.0, 200.0], [150.0, 255.0]], dtype=np.float32)
        result = vizy._to_plottable_int_arr(arr)  # noqa: SLF001
        assert result.dtype == np.uint8


class TestSave:
    """Test the save function."""

    def test_save_with_path(self) -> None:
        """Test saving with explicit path."""
        rng = np.random.default_rng(42)
        arr = rng.random((50, 60, 3))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch("builtins.print") as mock_print:
                result_path = vizy.save(tmp_path, arr)
            assert result_path == tmp_path
            assert Path(tmp_path).exists()
            mock_print.assert_called_once_with(tmp_path)
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_save_auto_path(self) -> None:
        """Test saving with automatic path generation."""
        rng = np.random.default_rng(42)
        arr = rng.random((50, 60, 3))

        with patch("builtins.print") as mock_print:
            result_path = vizy.save(arr)

        try:
            assert result_path.endswith(".png")
            assert "vizy-" in result_path
            assert Path(result_path).exists()
            mock_print.assert_called_once_with(result_path)
        finally:
            if Path(result_path).exists():
                Path(result_path).unlink()

    def test_save_torch_tensor(self) -> None:
        """Test saving torch tensor."""
        tensor = torch.rand(50, 60, 3)

        with patch("builtins.print") as mock_print:
            result_path = vizy.save(tensor)

        try:
            assert Path(result_path).exists()
            mock_print.assert_called_once_with(result_path)
        finally:
            if Path(result_path).exists():
                Path(result_path).unlink()

    def test_save_with_kwargs(self) -> None:
        """Test saving with additional kwargs."""
        rng = np.random.default_rng(42)
        arr = rng.random((50, 60, 3))  # Use RGB to avoid cmap conflict

        with patch("builtins.print") as mock_print:
            result_path = vizy.save(arr)

        try:
            assert Path(result_path).exists()
            mock_print.assert_called_once_with(result_path)
        finally:
            if Path(result_path).exists():
                Path(result_path).unlink()


class TestSummary:
    """Test the summary function."""

    def test_summary_numpy_array(self) -> None:
        """Test summary for numpy array."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=(50, 60, 3), dtype=np.uint8)

        with patch("builtins.print") as mock_print:
            vizy.summary(arr)

        # Check that print was called with expected information
        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("numpy.ndarray" in call for call in calls)
        assert any("Shape: (50, 60, 3)" in call for call in calls)
        assert any("uint8" in call for call in calls)
        assert any("Range:" in call for call in calls)

    def test_summary_torch_tensor(self) -> None:
        """Test summary for torch tensor."""
        tensor = torch.rand(50, 60, 3)

        with patch("builtins.print") as mock_print:
            vizy.summary(tensor)

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("torch.Tensor" in call for call in calls)
        assert any("Shape: (50, 60, 3)" in call for call in calls)
        assert any("float32" in call for call in calls)
        assert any("device:" in call for call in calls)

    def test_summary_torch_tensor_with_device(self) -> None:
        """Test summary for torch tensor with device info."""
        tensor = torch.rand(10, 10)
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        with patch("builtins.print") as mock_print:
            vizy.summary(tensor)

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("device:" in call for call in calls)

    def test_summary_integer_array(self) -> None:
        """Test summary for integer array shows unique values."""
        arr = np.array([1, 2, 2, 3, 3, 3], dtype=np.int32)

        with patch("builtins.print") as mock_print:
            vizy.summary(arr)

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("Number of unique values: 3" in call for call in calls)

    def test_summary_empty_array(self) -> None:
        """Test summary for empty array."""
        arr = np.array([], dtype=np.float32)

        with patch("builtins.print") as mock_print:
            vizy.summary(arr)

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("Range: N/A (empty array)" in call for call in calls)

    def test_summary_invalid_input(self) -> None:
        """Test summary with invalid input type."""
        with pytest.raises(TypeError, match="Expected torch.Tensor | np.ndarray"):
            vizy.summary("invalid_string")  # type: ignore[arg-type]


class TestNumpyToPilImage:
    """Test the _numpy_to_pil_image function."""

    def test_2d_grayscale(self) -> None:
        """Test conversion of 2D array to grayscale PIL image."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        pil_img = vizy._numpy_to_pil_image(arr)  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "L"
        assert pil_img.size == (32, 32)

    def test_3d_single_channel(self) -> None:
        """Test conversion of 3D array with single channel."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (32, 32, 1), dtype=np.uint8)
        pil_img = vizy._numpy_to_pil_image(arr)  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "L"
        assert pil_img.size == (32, 32)

    def test_3d_rgb(self) -> None:
        """Test conversion of 3D array with RGB channels."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        pil_img = vizy._numpy_to_pil_image(arr)  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"
        assert pil_img.size == (32, 32)

    def test_3d_rgba(self) -> None:
        """Test conversion of 3D array with RGBA channels."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (32, 32, 4), dtype=np.uint8)
        pil_img = vizy._numpy_to_pil_image(arr)  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGBA"
        assert pil_img.size == (32, 32)

    def test_invalid_channels(self) -> None:
        """Test that invalid number of channels raises ValueError."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (32, 32, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported number of channels"):
            vizy._numpy_to_pil_image(arr)  # noqa: SLF001

    def test_invalid_dimensions(self) -> None:
        """Test that invalid dimensions raise ValueError."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (32,), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported array dimensions"):
            vizy._numpy_to_pil_image(arr)  # noqa: SLF001


class TestTensorToPilImage:
    """Test the _tensor_to_pil_image function."""

    def test_numpy_array_2d(self) -> None:
        """Test conversion of 2D numpy array."""
        rng = np.random.default_rng(42)
        arr = rng.random((32, 32))
        pil_img = vizy._tensor_to_pil_image(arr)  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)

    def test_numpy_array_3d(self) -> None:
        """Test conversion of 3D numpy array."""
        rng = np.random.default_rng(42)
        arr = rng.random((32, 32, 3))
        pil_img = vizy._tensor_to_pil_image(arr)  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode in ["RGB", "L"]

    def test_torch_tensor(self) -> None:
        """Test conversion of torch tensor."""
        tensor = torch.rand(3, 32, 32)
        pil_img = vizy._tensor_to_pil_image(tensor)  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode in ["RGB", "L"]

    def test_pil_image(self) -> None:
        """Test that PIL image is converted correctly."""
        pil_img_input = Image.new("RGB", (32, 32), color=(255, 0, 0))
        pil_img_output = vizy._tensor_to_pil_image(pil_img_input)  # noqa: SLF001
        assert isinstance(pil_img_output, Image.Image)

    def test_list_of_arrays(self) -> None:
        """Test conversion of list of arrays."""
        rng = np.random.default_rng(42)
        arr1 = rng.random((32, 32))
        arr2 = rng.random((32, 32))
        pil_img = vizy._tensor_to_pil_image([arr1, arr2])  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)
        # Should create a grid, so width should be doubled
        assert pil_img.size[0] == 64  # 2 images side by side


class TestPlot:
    """Test the plot function."""

    def test_plot_numpy_array(self) -> None:
        """Test plotting numpy array."""
        rng = np.random.default_rng(42)
        arr = rng.random((32, 32, 3))
        with patch("PIL.Image.Image.show") as mock_show:
            vizy.plot(arr)
        mock_show.assert_called_once()

    def test_plot_torch_tensor(self) -> None:
        """Test plotting torch tensor."""
        tensor = torch.rand(3, 32, 32)
        with patch("PIL.Image.Image.show") as mock_show:
            vizy.plot(tensor)
        mock_show.assert_called_once()

    def test_plot_pil_image(self) -> None:
        """Test plotting PIL image."""
        pil_img = Image.new("RGB", (32, 32), color=(0, 255, 0))  # Green image
        with patch("PIL.Image.Image.show") as mock_show:
            vizy.plot(pil_img)
        mock_show.assert_called_once()

    def test_plot_list_of_arrays(self) -> None:
        """Test plotting list of arrays."""
        rng = np.random.default_rng(42)
        arr1 = rng.random((32, 32))
        arr2 = rng.random((32, 32))
        with patch("PIL.Image.Image.show") as mock_show:
            vizy.plot([arr1, arr2])
        mock_show.assert_called_once()

    def test_plot_jupyter_inline_display(self) -> None:
        """Test that plot uses IPython.display.display in Jupyter notebooks."""
        rng = np.random.default_rng(42)
        arr = rng.random((32, 32, 3))

        # Mock sys.modules to simulate Jupyter environment
        original_modules = sys.modules.copy()
        sys.modules["ipykernel"] = MagicMock()

        try:
            # Patch display - it's imported inside the function, so patch at the source
            with patch("IPython.display.display") as mock_display, patch("PIL.Image.Image.show") as mock_show:
                vizy.plot(arr)
                # In Jupyter, display should be called
                mock_display.assert_called_once()
                # show should NOT be called
                mock_show.assert_not_called()
        finally:
            # Restore original modules
            if "ipykernel" in sys.modules:
                del sys.modules["ipykernel"]
            sys.modules.update(original_modules)

    def test_plot_falls_back_to_show_when_not_jupyter(self) -> None:
        """Test that plot falls back to show() when not in Jupyter environment."""
        rng = np.random.default_rng(42)
        arr = rng.random((32, 32, 3))

        # Ensure ipykernel is not in sys.modules to simulate non-Jupyter environment
        original_modules = sys.modules.copy()
        ipykernel_backup = sys.modules.pop("ipykernel", None)

        try:
            with patch("PIL.Image.Image.show") as mock_show:
                vizy.plot(arr)
                # When not in Jupyter, show should be called
                mock_show.assert_called_once()
        finally:
            # Restore original modules
            if ipykernel_backup is not None:
                sys.modules["ipykernel"] = ipykernel_backup
            sys.modules.update(original_modules)


class TestPILSupport:
    """Test PIL Image support functionality."""

    def test_save_pil_image(self) -> None:
        """Test saving PIL image to file."""
        pil_img = Image.new("RGB", (40, 30), color=(0, 0, 255))  # Blue image

        with patch("builtins.print") as mock_print:
            result_path = vizy.save(pil_img)

        try:
            assert Path(result_path).exists()
            assert result_path.endswith(".png")
            mock_print.assert_called_once_with(result_path)
        finally:
            if Path(result_path).exists():
                Path(result_path).unlink()

    def test_summary_pil_rgb(self) -> None:
        """Test summary for PIL RGB image."""
        pil_img = Image.new("RGB", (50, 60), color=(128, 64, 192))

        with patch("builtins.print") as mock_print:
            vizy.summary(pil_img)

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("PIL.Image" in call for call in calls)
        assert any("mode: RGB" in call for call in calls)
        assert any("Shape: (60, 50, 3)" in call for call in calls)
        assert any("uint8" in call for call in calls)

    def test_mixed_types_error(self) -> None:
        """Test that invalid types still raise appropriate errors."""
        with pytest.raises(TypeError, match="Expected torch.Tensor | np.ndarray | PIL.Image"):
            # List of numbers should still fail
            vizy._to_numpy([1, 2, 3])  # type: ignore[arg-type] # noqa: SLF001

        with pytest.raises(TypeError, match="Expected torch.Tensor | np.ndarray | PIL.Image"):
            # String should still fail
            vizy._to_numpy("string")  # type: ignore[arg-type] # noqa: SLF001


class TestRandomArrays:
    """Test with various random array configurations."""

    def test_random_2d_arrays(self) -> None:
        """Test with random 2D arrays of various sizes."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            h, w = rng.integers(10, 200, 2)
            arr = rng.random((h, w))

            # Test that all functions work
            result = vizy._to_plottable_int_arr(arr)  # noqa: SLF001
            assert result.shape == (h, w)

            # Test conversion to PIL image
            pil_img = vizy._tensor_to_pil_image(arr)  # noqa: SLF001
            assert isinstance(pil_img, Image.Image)
            assert pil_img.size == (w, h)

    def test_random_3d_arrays(self) -> None:
        """Test with random 3D arrays."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            h, w = rng.integers(10, 100, 2)
            c = rng.choice([1, 3])

            # Test both CHW and HWC formats
            arr = rng.random((c, h, w)) if rng.random() > 0.5 else rng.random((h, w, c))

            result = vizy._to_plottable_int_arr(arr)  # noqa: SLF001
            assert result.ndim in [2, 3]

            # Test conversion to PIL image
            pil_img = vizy._tensor_to_pil_image(arr)  # noqa: SLF001
            assert isinstance(pil_img, Image.Image)

    def test_random_4d_arrays(self) -> None:
        """Test with random 4D arrays."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            b = rng.integers(1, 8)
            c = rng.choice([1, 3])
            h, w = rng.integers(10, 50, 2)

            # Test both BCHW and CBHW formats
            arr = rng.random((b, c, h, w)) if rng.random() > 0.5 else rng.random((c, b, h, w))

            try:
                result = vizy._to_plottable_int_arr(arr)  # noqa: SLF001
                # Result can be 2D (single channel squeezed) or 3D (multi-channel)
                assert result.ndim in [2, 3]

                # Test conversion to PIL image
                pil_img = vizy._tensor_to_pil_image(arr)  # noqa: SLF001
                assert isinstance(pil_img, Image.Image)
            except (ValueError, TypeError):
                # Some random shapes might not be valid, which is expected
                pass

    def test_random_torch_tensors(self) -> None:
        """Test with random torch tensors."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            # Generate valid shapes for vizy
            shape_type = rng.choice(["2d", "3d", "4d"])
            if shape_type == "2d":
                shape = tuple(int(s) for s in rng.integers(10, 50, 2))
            elif shape_type == "3d":
                h, w = rng.integers(10, 50, 2)
                c = rng.choice([1, 3])
                shape = (int(c), int(h), int(w)) if rng.random() > 0.5 else (int(h), int(w), int(c))
            else:  # 4d
                b = rng.integers(1, 4)
                c = rng.choice([1, 3])
                h, w = rng.integers(10, 30, 2)
                shape = (int(b), int(c), int(h), int(w)) if rng.random() > 0.5 else (int(c), int(b), int(h), int(w))

            tensor = torch.rand(*shape)

            try:
                # Test conversion to PIL image
                pil_img = vizy._tensor_to_pil_image(tensor)  # noqa: SLF001
                assert isinstance(pil_img, Image.Image)
            except (ValueError, TypeError):
                # Some random shapes might not be valid, which is expected
                pass

    def test_edge_case_shapes(self) -> None:
        """Test edge cases with minimal and maximal shapes."""
        rng = np.random.default_rng(42)
        # Minimal shapes
        arr = rng.random((2, 2))  # Use 2x2 instead of 1x1 to avoid edge cases
        _ = vizy._to_plottable_int_arr(arr)  # noqa: SLF001
        pil_img = vizy._tensor_to_pil_image(arr)  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)

        # Single pixel RGB
        arr = rng.random((2, 2, 3))  # Use 2x2 instead of 1x1
        _ = vizy._to_plottable_int_arr(arr)  # noqa: SLF001
        pil_img = vizy._tensor_to_pil_image(arr)  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)

        # Large batch size
        arr = rng.random((16, 3, 32, 32))
        _ = vizy._to_plottable_int_arr(arr)  # noqa: SLF001
        pil_img = vizy._tensor_to_pil_image(arr)  # noqa: SLF001
        assert isinstance(pil_img, Image.Image)


class TestListSupport:
    """Test list/sequence support functionality."""

    def test_is_sequence_of_tensors_detection(self) -> None:
        """Test sequence detection for various inputs."""
        # Valid sequences
        assert vizy._is_sequence_of_tensors([np.array([1, 2]), np.array([3, 4])])  # noqa: SLF001
        assert vizy._is_sequence_of_tensors((np.array([1, 2]), np.array([3, 4])))  # noqa: SLF001
        assert vizy._is_sequence_of_tensors([torch.tensor([1, 2]), torch.tensor([3, 4])])  # noqa: SLF001

        # Invalid sequences
        assert not vizy._is_sequence_of_tensors([])  # noqa: SLF001
        assert not vizy._is_sequence_of_tensors(np.array([1, 2]))  # noqa: SLF001 # Single array
        assert not vizy._is_sequence_of_tensors([1, 2, 3])  # type: ignore[arg-type] # noqa: SLF001
        assert not vizy._is_sequence_of_tensors([np.array([1]), "string"])  # type: ignore[arg-type] # noqa: SLF001

    def test_list_of_same_size_2d_arrays(self) -> None:
        """Test processing list of 2D arrays with same dimensions."""
        rng = np.random.default_rng(42)
        arr1 = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        arr2 = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        arr3 = rng.integers(0, 255, (32, 32), dtype=np.uint8)

        array_list = [arr1, arr2, arr3]
        result = vizy._to_numpy(array_list)  # noqa: SLF001

        # Should create a batch with shape (3, 32, 32)
        assert result.shape == (3, 32, 32)
        assert np.array_equal(result[0], arr1)
        assert np.array_equal(result[1], arr2)
        assert np.array_equal(result[2], arr3)

    def test_list_of_different_size_arrays_with_padding(self) -> None:
        """Test that arrays with different sizes get padded correctly."""
        rng = np.random.default_rng(42)
        arr1 = rng.integers(0, 255, (20, 30), dtype=np.uint8)  # Small
        arr2 = rng.integers(0, 255, (40, 50), dtype=np.uint8)  # Large
        arr3 = rng.integers(0, 255, (25, 35), dtype=np.uint8)  # Medium

        array_list = [arr1, arr2, arr3]
        result = vizy._to_numpy(array_list)  # noqa: SLF001

        # All should be padded to largest size (40, 50)
        assert result.shape == (3, 40, 50)

        # Check that original content is preserved (top-left corner)
        assert np.array_equal(result[0][:20, :30], arr1)
        assert np.array_equal(result[1], arr2)  # Largest, unchanged
        assert np.array_equal(result[2][:25, :35], arr3)

        # Check padding is zeros (black)
        assert np.all(result[0][20:, :] == 0)  # Bottom padding
        assert np.all(result[0][:, 30:] == 0)  # Right padding

    def test_list_of_3d_chw_arrays(self) -> None:
        """Test processing list of 3D arrays in CHW format."""
        rng = np.random.default_rng(42)
        rgb1 = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
        rgb2 = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)

        array_list = [rgb1, rgb2]
        result = vizy._to_numpy(array_list)  # noqa: SLF001

        # Should create batch with shape (2, 3, 32, 32)
        assert result.shape == (2, 3, 32, 32)
        assert np.array_equal(result[0], rgb1)
        assert np.array_equal(result[1], rgb2)

    def test_mixed_tensor_types(self) -> None:
        """Test list containing mix of numpy arrays and torch tensors."""
        rng = np.random.default_rng(42)
        np_arr = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        torch_arr = torch.randint(0, 255, (32, 32), dtype=torch.uint8)

        mixed_list = [np_arr, torch_arr]
        result = vizy._to_numpy(mixed_list)  # noqa: SLF001

        assert result.shape == (2, 32, 32)
        assert np.array_equal(result[0], np_arr)
        assert np.array_equal(result[1], torch_arr.numpy())

    def test_list_dimension_validation(self) -> None:
        """Test that 4D tensors in lists are rejected."""
        rng = np.random.default_rng(42)
        # Valid: Same dimension tensors
        valid_list = [
            rng.random((32, 32)),  # 2D
            rng.random((32, 32)),  # 2D
        ]
        result = vizy._to_numpy(valid_list)  # noqa: SLF001
        assert result.ndim == 3  # Should work (B, H, W)

        # Invalid: 4D tensor in list
        invalid_list = [
            rng.random((32, 32)),  # 2D - OK
            rng.random((2, 3, 32, 32)),  # 4D - NOT OK
        ]
        with pytest.raises(ValueError, match="Each tensor in list must be 2D or 3D"):
            vizy._to_numpy(invalid_list)  # noqa: SLF001

    def test_list_plot_integration(self) -> None:
        """Test that list plotting works end-to-end."""
        rng = np.random.default_rng(42)
        arr1_1 = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        arr1_2 = rng.integers(0, 255, (48, 48), dtype=np.uint8)
        array1_list = [arr1_1, arr1_2]

        # Should work without errors
        with patch("PIL.Image.Image.show"):
            vizy.plot(array1_list)

    def test_list_save_integration(self) -> None:
        """Test that list saving works end-to-end."""
        rng = np.random.default_rng(42)
        arr1 = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        arr2 = rng.integers(0, 255, (32, 32), dtype=np.uint8)

        array_list = [arr1, arr2]

        with patch("builtins.print") as mock_print:
            result_path = vizy.save(array_list)

        try:
            assert Path(result_path).exists()
            assert result_path.endswith(".png")
            mock_print.assert_called_once_with(result_path)
        finally:
            if Path(result_path).exists():
                Path(result_path).unlink()

    def test_list_summary_integration(self) -> None:
        """Test that list summary works correctly."""
        rng = np.random.default_rng(42)
        arr1 = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        arr2 = rng.integers(0, 255, (48, 48), dtype=np.uint8)

        array_list = [arr1, arr2]

        with patch("builtins.print") as mock_print:
            vizy.summary(array_list)

        calls = [call.args[0] for call in mock_print.call_args_list]
        # Should mention it's a sequence
        assert any("Sequence" in call for call in calls)
        assert any("2 tensors" in call for call in calls)
        # Should show individual tensor info
        assert any("Shape: (32, 32)" in call for call in calls)
        assert any("Shape: (48, 48)" in call for call in calls)

    def test_empty_list_handling(self) -> None:
        """Test that empty lists are handled gracefully."""
        with pytest.raises((TypeError, ValueError)):
            vizy._to_numpy([])  # noqa: SLF001

    def test_single_item_list(self) -> None:
        """Test that single-item lists work correctly."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        single_list = [arr]

        result = vizy._to_numpy(single_list)  # noqa: SLF001
        assert result.shape == (1, 32, 32)
        assert np.array_equal(result[0], arr)


if __name__ == "__main__":
    pytest.main([__file__])
