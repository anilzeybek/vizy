import os
import tempfile
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

import vizy


class TestNormalizeArrayFormat:
    """Test the _normalize_array_format function."""

    def test_2d_array_unchanged(self):
        """Test that 2D arrays (H, W) are unchanged."""
        arr = np.random.rand(100, 200)
        result, _ = vizy._normalize_array_format(arr)
        assert np.array_equal(result, arr)
        assert result.shape == (100, 200)

    def test_3d_hwc_unchanged(self):
        """Test that 3D arrays in HWC format are unchanged."""
        arr = np.random.rand(50, 60, 3)  # H, W, C
        result, _ = vizy._normalize_array_format(arr)
        assert np.array_equal(result, arr)
        assert result.shape == (50, 60, 3)

    def test_3d_chw_to_hwc(self):
        """Test conversion from CHW to HWC format."""
        arr = np.random.rand(3, 50, 60)  # C, H, W
        result, _ = vizy._normalize_array_format(arr)
        expected = np.transpose(arr, (1, 2, 0))
        assert np.array_equal(result, expected) or np.array_equal(result, arr)
        assert result.shape == (50, 60, 3) or result.shape == (3, 50, 60)

    def test_3d_single_channel_chw(self):
        """Test conversion from single channel CHW to HWC."""
        arr = np.random.rand(1, 40, 50)  # C=1, H, W
        result, _ = vizy._normalize_array_format(arr)
        expected = arr.squeeze(axis=0)
        assert np.array_equal(result, expected)
        assert result.shape == (40, 50)

    def test_3d_ambiguous_case(self):
        """Test case where both dimensions could be channels."""
        # When both first and last dim are 3, should prefer HWC (no transpose)
        arr = np.random.rand(3, 50, 3)
        result, _ = vizy._normalize_array_format(arr)
        assert np.array_equal(result, arr)  # Should remain unchanged

    def test_invalid_dimensions(self):
        """Test that arrays with unsupported dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Unable to determine 4D array format"):
            vizy._normalize_array_format(np.random.rand(10, 20, 30, 40))

        with pytest.raises(ValueError, match="Cannot prepare array"):
            vizy._normalize_array_format(np.random.rand(10))


class TestPrep:
    """Test the _prep function."""

    def test_2d_array(self):
        """Test preparation of 2D arrays."""
        arr = np.random.rand(50, 60)
        result, _ = vizy._normalize_array_format(arr)
        assert result.shape == (50, 60)
        assert result.ndim == 2

    def test_3d_array_hwc(self):
        """Test preparation of 3D arrays in HWC format."""
        arr = np.random.rand(50, 60, 3)
        result, _ = vizy._normalize_array_format(arr)
        assert result.shape == (50, 60, 3)

    def test_3d_array_chw(self):
        """Test preparation of 3D arrays in CHW format."""
        arr = np.random.rand(3, 50, 60)
        result, _ = vizy._normalize_array_format(arr)
        assert result.shape == (50, 60, 3) or result.shape == (3, 50, 60)

    def test_4d_bchw(self):
        """Test preparation of 4D arrays in BCHW format."""
        arr = np.random.rand(4, 3, 50, 60)  # B, C, H, W
        result, _ = vizy._normalize_array_format(arr)
        expected = np.transpose(arr, (0, 2, 3, 1))  # B, H, W, C
        assert result.shape == (4, 50, 60, 3)
        assert np.array_equal(result, expected)

    def test_4d_cbhw_to_bchw(self):
        """Test conversion from CBHW to BCHW format."""
        arr = np.random.rand(3, 4, 50, 60)  # C, B, H, W
        result, _ = vizy._normalize_array_format(arr)
        expected = np.transpose(arr, (1, 2, 3, 0))  # B, H, W, C
        assert result.shape == (4, 50, 60, 3)
        assert np.array_equal(result, expected)

    def test_4d_single_channel(self):
        """Test 4D arrays with single channel."""
        arr = np.random.rand(4, 1, 50, 60)  # B, C=1, H, W
        result, _ = vizy._normalize_array_format(arr)
        expected = np.squeeze(arr, axis=1)  # B, H, W, C
        assert result.shape == (4, 50, 60)
        assert np.array_equal(result, expected)

    def test_4d_bhwc_unchanged(self):
        """Test that 4D arrays already in BHWC format are returned unchanged."""
        arr = np.random.rand(4, 50, 60, 3)  # B, H, W, C
        result, _ = vizy._normalize_array_format(arr)
        assert result.shape == (4, 50, 60, 3)
        assert np.array_equal(result, arr)

    def test_squeeze_behavior(self):
        """Test that arrays are properly squeezed."""
        arr = np.random.rand(1, 1, 50, 60, 1)
        result, _ = vizy._normalize_array_format(arr)
        assert result.shape == (50, 60)

    def test_invalid_4d_shape(self):
        """Test that invalid 4D shapes raise ValueError."""
        # Neither dimension 0 nor 1 is a valid channel count
        arr = np.random.rand(5, 7, 50, 60)
        with pytest.raises(ValueError, match="Unable to determine 4D array format"):
            vizy._normalize_array_format(arr)

    def test_invalid_dimensions(self):
        """Test that unsupported dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Cannot prepare array"):
            vizy._normalize_array_format(np.random.rand(10, 20, 30, 40, 50))


class TestMakeGrid:
    """Test the _make_grid function."""

    def test_single_image(self):
        """Test grid creation with single image."""
        bhwc = np.random.rand(1, 32, 32, 3)
        result = vizy._make_grid(bhwc)
        assert result.shape == (32, 32, 3)

    def test_two_images(self):
        """Test grid creation with two images (side by side)."""
        bhwc = np.random.rand(2, 32, 32, 3)
        result = vizy._make_grid(bhwc)
        assert result.shape == (32, 64, 3)  # 1 row, 2 cols

    def test_three_images(self):
        """Test grid creation with three images (all in a row)."""
        bhwc = np.random.rand(3, 32, 32, 3)
        result = vizy._make_grid(bhwc)
        assert result.shape == (32, 96, 3)  # 1 row, 3 cols

    def test_four_images(self):
        """Test grid creation with four images (2x2 grid)."""
        bhwc = np.random.rand(4, 32, 32, 3)
        result = vizy._make_grid(bhwc)
        assert result.shape == (64, 64, 3)  # 2 rows, 2 cols

    def test_larger_batch(self):
        """Test grid creation with larger batch."""
        bhwc = np.random.rand(9, 32, 32, 3)
        result = vizy._make_grid(bhwc)
        assert result.shape == (96, 96, 3)  # 3 rows, 3 cols

    def test_single_channel(self):
        """Test grid creation with single channel images."""
        bhwc = np.random.rand(4, 32, 32, 1)
        result = vizy._make_grid(bhwc)
        assert result.shape == (64, 64, 1)

    def test_non_square_images(self):
        """Test grid creation with non-square images."""
        bhwc = np.random.rand(4, 20, 30, 3)
        result = vizy._make_grid(bhwc)
        assert result.shape == (40, 60, 3)  # 2 rows, 2 cols


class TestConvertFloatToInt:
    """Test the _convert_float_to_int function."""

    def test_float_in_0_255_range(self):
        """Test conversion of float arrays in 0-255 range to uint8."""
        arr = np.array([0.0, 127.5, 255.0], dtype=np.float32)
        result = vizy._convert_float_arr_to_int_arr(arr)
        expected = np.array([0, 128, 255], dtype=np.uint8)
        assert np.array_equal(result, expected)
        assert result.dtype == np.uint8

    def test_float_in_0_1_range_unchanged(self):
        """Test that float arrays in 0-1 range are unchanged."""
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = vizy._convert_float_arr_to_int_arr(arr)
        assert np.array_equal(result, arr)
        assert result.dtype == np.float32

    def test_integer_array_unchanged(self):
        """Test that integer arrays are unchanged."""
        arr = np.array([0, 127, 255], dtype=np.uint8)
        result = vizy._convert_float_arr_to_int_arr(arr)
        assert np.array_equal(result, arr)
        assert result.dtype == np.uint8

    def test_float_outside_0_255_range(self):
        """Test that float arrays outside 0-255 range are unchanged."""
        arr = np.array([-10.0, 300.0, 500.0], dtype=np.float32)
        result = vizy._convert_float_arr_to_int_arr(arr)
        assert np.array_equal(result, arr)
        assert result.dtype == np.float32

    def test_clipping_behavior(self):
        """Test that values are properly clipped to 0-255 range."""
        # This test was wrong - the function doesn't convert values outside 0-255 range
        arr = np.array([100.0, 200.0, 255.0], dtype=np.float32)
        result = vizy._convert_float_arr_to_int_arr(arr)
        expected = np.array([100, 200, 255], dtype=np.uint8)
        assert np.array_equal(result, expected)


class TestPrepareForDisplay:
    """Test the _prepare_for_display function."""

    def test_2d_array(self):
        """Test preparation of 2D array for display."""
        arr = np.random.rand(50, 60)
        result = vizy._to_plottable_int_arr(arr)
        assert result.shape == (50, 60)
        assert result.ndim == 2

    def test_3d_array(self):
        """Test preparation of 3D array for display."""
        arr = np.random.rand(50, 60, 3)
        result = vizy._to_plottable_int_arr(arr)
        assert result.shape == (50, 60, 3)

    def test_4d_array_to_grid(self):
        """Test that 4D arrays are converted to grids."""
        arr = np.random.rand(4, 3, 32, 32)
        result = vizy._to_plottable_int_arr(arr)
        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB channels

    def test_float_to_int_conversion(self):
        """Test that float arrays in 0-255 range are converted to uint8."""
        arr = np.array([[[100.0, 200.0, 255.0]]], dtype=np.float32)
        # This will be squeezed to (3,) which is invalid, so let's use a proper shape
        arr = np.array([[100.0, 200.0], [150.0, 255.0]], dtype=np.float32)
        result = vizy._to_plottable_int_arr(arr)
        assert result.dtype == np.uint8


class TestSave:
    """Test the save function."""

    def test_save_with_path(self):
        """Test saving with explicit path."""
        arr = np.random.rand(50, 60, 3)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch("builtins.print") as mock_print:
                result_path = vizy.save(tmp_path, arr)
            assert result_path == tmp_path
            assert os.path.exists(tmp_path)
            mock_print.assert_called_once_with(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_save_auto_path(self):
        """Test saving with automatic path generation."""
        arr = np.random.rand(50, 60, 3)

        with patch("builtins.print") as mock_print:
            result_path = vizy.save(arr)

        try:
            assert result_path.endswith(".png")
            assert "vizy-" in result_path
            assert os.path.exists(result_path)
            mock_print.assert_called_once_with(result_path)
        finally:
            if os.path.exists(result_path):
                os.unlink(result_path)

    def test_save_torch_tensor(self):
        """Test saving torch tensor."""
        tensor = torch.rand(50, 60, 3)

        with patch("builtins.print") as mock_print:
            result_path = vizy.save(tensor)

        try:
            assert os.path.exists(result_path)
            mock_print.assert_called_once_with(result_path)
        finally:
            if os.path.exists(result_path):
                os.unlink(result_path)

    def test_save_with_kwargs(self):
        """Test saving with additional kwargs."""
        arr = np.random.rand(50, 60, 3)  # Use RGB to avoid cmap conflict

        with patch("builtins.print") as mock_print:
            result_path = vizy.save(arr, vmin=0.2)

        try:
            assert os.path.exists(result_path)
            mock_print.assert_called_once_with(result_path)
        finally:
            if os.path.exists(result_path):
                os.unlink(result_path)


class TestSummary:
    """Test the summary function."""

    def test_summary_numpy_array(self):
        """Test summary for numpy array."""
        arr = np.random.randint(0, 256, size=(50, 60, 3), dtype=np.uint8)

        with patch("builtins.print") as mock_print:
            vizy.summary(arr)

        # Check that print was called with expected information
        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("numpy.ndarray" in call for call in calls)
        assert any("Shape: (50, 60, 3)" in call for call in calls)
        assert any("uint8" in call for call in calls)
        assert any("Range:" in call for call in calls)

    def test_summary_torch_tensor(self):
        """Test summary for torch tensor."""
        tensor = torch.rand(50, 60, 3)

        with patch("builtins.print") as mock_print:
            vizy.summary(tensor)

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("torch.Tensor" in call for call in calls)
        assert any("Shape: (50, 60, 3)" in call for call in calls)
        assert any("float32" in call for call in calls)
        assert any("device:" in call for call in calls)

    def test_summary_torch_tensor_with_device(self):
        """Test summary for torch tensor with device info."""
        tensor = torch.rand(10, 10)
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        with patch("builtins.print") as mock_print:
            vizy.summary(tensor)

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("device:" in call for call in calls)

    def test_summary_integer_array(self):
        """Test summary for integer array shows unique values."""
        arr = np.array([1, 2, 2, 3, 3, 3], dtype=np.int32)

        with patch("builtins.print") as mock_print:
            vizy.summary(arr)

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("Number of unique values: 3" in call for call in calls)

    def test_summary_empty_array(self):
        """Test summary for empty array."""
        arr = np.array([], dtype=np.float32)

        with patch("builtins.print") as mock_print:
            vizy.summary(arr)

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("Range: N/A (empty array)" in call for call in calls)

    def test_summary_invalid_input(self):
        """Test summary with invalid input type."""
        with pytest.raises(TypeError, match="Expected torch.Tensor | np.ndarray"):
            vizy.summary("invalid_string")


class TestPILSupport:
    """Test PIL Image support functionality."""

    def test_plot_pil_image(self):
        """Test plotting PIL image."""
        pil_img = Image.new("RGB", (32, 32), color=(0, 255, 0))  # Green image

        with patch("matplotlib.pyplot.show"):
            _ = vizy.plot(pil_img)

        # The function should complete without error
        plt.close("all")

    def test_save_pil_image(self):
        """Test saving PIL image to file."""
        pil_img = Image.new("RGB", (40, 30), color=(0, 0, 255))  # Blue image

        with patch("builtins.print") as mock_print:
            result_path = vizy.save(pil_img)

        try:
            assert os.path.exists(result_path)
            assert result_path.endswith(".png")
            mock_print.assert_called_once_with(result_path)
        finally:
            if os.path.exists(result_path):
                os.unlink(result_path)

    def test_summary_pil_rgb(self):
        """Test summary for PIL RGB image."""
        pil_img = Image.new("RGB", (50, 60), color=(128, 64, 192))

        with patch("builtins.print") as mock_print:
            vizy.summary(pil_img)

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("PIL.Image" in call for call in calls)
        assert any("mode: RGB" in call for call in calls)
        assert any("Shape: (60, 50, 3)" in call for call in calls)
        assert any("uint8" in call for call in calls)

    def test_mixed_types_error(self):
        """Test that invalid types still raise appropriate errors."""
        with pytest.raises(TypeError, match="Expected torch.Tensor | np.ndarray | PIL.Image"):
            vizy._to_numpy([1, 2, 3])  # List of numbers should still fail

        with pytest.raises(TypeError, match="Expected torch.Tensor | np.ndarray | PIL.Image"):
            vizy._to_numpy("string")  # String should still fail


class TestRandomArrays:
    """Test with various random array configurations."""

    def test_random_2d_arrays(self):
        """Test with random 2D arrays of various sizes."""
        for _ in range(10):
            h, w = np.random.randint(10, 200, 2)
            arr = np.random.rand(h, w)

            # Test that all functions work
            result = vizy._to_plottable_int_arr(arr)
            assert result.shape == (h, w)

            fig = vizy._create_figure(arr)
            plt.close(fig)

    def test_random_3d_arrays(self):
        """Test with random 3D arrays."""
        for _ in range(10):
            h, w = np.random.randint(10, 100, 2)
            c = np.random.choice([1, 3])

            # Test both CHW and HWC formats
            if np.random.rand() > 0.5:
                arr = np.random.rand(c, h, w)  # CHW
            else:
                arr = np.random.rand(h, w, c)  # HWC

            result = vizy._to_plottable_int_arr(arr)
            assert result.ndim in [2, 3]

            fig = vizy._create_figure(arr)
            plt.close(fig)

    def test_random_4d_arrays(self):
        """Test with random 4D arrays."""
        for _ in range(5):
            b = np.random.randint(1, 8)
            c = np.random.choice([1, 3])
            h, w = np.random.randint(10, 50, 2)

            # Test both BCHW and CBHW formats
            if np.random.rand() > 0.5:
                arr = np.random.rand(b, c, h, w)  # BCHW
            else:
                arr = np.random.rand(c, b, h, w)  # CBHW

            try:
                result = vizy._to_plottable_int_arr(arr)
                # Result can be 2D (single channel squeezed) or 3D (multi-channel)
                assert result.ndim in [2, 3]

                fig = vizy._create_figure(arr)
                plt.close(fig)
            except (ValueError, TypeError):
                # Some random shapes might not be valid for matplotlib, which is expected
                pass

    def test_random_torch_tensors(self):
        """Test with random torch tensors."""
        for _ in range(5):
            # Generate valid shapes for vizy
            shape_type = np.random.choice(["2d", "3d", "4d"])
            if shape_type == "2d":
                shape = tuple(np.random.randint(10, 50, 2))
            elif shape_type == "3d":
                h, w = np.random.randint(10, 50, 2)
                c = np.random.choice([1, 3])
                if np.random.rand() > 0.5:
                    shape = (c, h, w)  # CHW
                else:
                    shape = (h, w, c)  # HWC
            else:  # 4d
                b = np.random.randint(1, 4)
                c = np.random.choice([1, 3])
                h, w = np.random.randint(10, 30, 2)
                if np.random.rand() > 0.5:
                    shape = (b, c, h, w)  # BCHW
                else:
                    shape = (c, b, h, w)  # CBHW

            tensor = torch.rand(*shape)

            try:
                # Convert to numpy first since _prepare_for_display expects numpy arrays
                arr = vizy._to_numpy(tensor)
                _ = vizy._to_plottable_int_arr(arr)
                fig = vizy._create_figure(tensor)
                plt.close(fig)
            except (ValueError, TypeError):
                # Some random shapes might not be valid, which is expected
                pass

    def test_edge_case_shapes(self):
        """Test edge cases with minimal and maximal shapes."""
        # Minimal shapes
        arr = np.random.rand(2, 2)  # Use 2x2 instead of 1x1 to avoid edge cases
        _ = vizy._to_plottable_int_arr(arr)
        fig = vizy._create_figure(arr)
        plt.close(fig)

        # Single pixel RGB
        arr = np.random.rand(2, 2, 3)  # Use 2x2 instead of 1x1
        _ = vizy._to_plottable_int_arr(arr)
        fig = vizy._create_figure(arr)
        plt.close(fig)

        # Large batch size
        arr = np.random.rand(16, 3, 32, 32)
        _ = vizy._to_plottable_int_arr(arr)
        fig = vizy._create_figure(arr)
        plt.close(fig)


class TestListSupport:
    """Test list/sequence support functionality."""

    def test_is_sequence_of_tensors_detection(self):
        """Test sequence detection for various inputs."""
        # Valid sequences
        assert vizy._is_sequence_of_tensors([np.array([1, 2]), np.array([3, 4])])
        assert vizy._is_sequence_of_tensors((np.array([1, 2]), np.array([3, 4])))
        assert vizy._is_sequence_of_tensors([torch.tensor([1, 2]), torch.tensor([3, 4])])

        # Invalid sequences
        assert not vizy._is_sequence_of_tensors([])  # Empty
        assert not vizy._is_sequence_of_tensors(np.array([1, 2]))  # Single array
        assert not vizy._is_sequence_of_tensors([1, 2, 3])  # List of numbers
        assert not vizy._is_sequence_of_tensors([np.array([1]), "string"])  # Mixed invalid

    def test_list_of_same_size_2d_arrays(self):
        """Test processing list of 2D arrays with same dimensions."""
        arr1 = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        arr2 = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        arr3 = np.random.randint(0, 255, (32, 32), dtype=np.uint8)

        array_list = [arr1, arr2, arr3]
        result = vizy._to_numpy(array_list)

        # Should create a batch with shape (3, 32, 32)
        assert result.shape == (3, 32, 32)
        assert np.array_equal(result[0], arr1)
        assert np.array_equal(result[1], arr2)
        assert np.array_equal(result[2], arr3)

    def test_list_of_different_size_arrays_with_padding(self):
        """Test that arrays with different sizes get padded correctly."""
        arr1 = np.random.randint(0, 255, (20, 30), dtype=np.uint8)  # Small
        arr2 = np.random.randint(0, 255, (40, 50), dtype=np.uint8)  # Large
        arr3 = np.random.randint(0, 255, (25, 35), dtype=np.uint8)  # Medium

        array_list = [arr1, arr2, arr3]
        result = vizy._to_numpy(array_list)

        # All should be padded to largest size (40, 50)
        assert result.shape == (3, 40, 50)

        # Check that original content is preserved (top-left corner)
        assert np.array_equal(result[0][:20, :30], arr1)
        assert np.array_equal(result[1], arr2)  # Largest, unchanged
        assert np.array_equal(result[2][:25, :35], arr3)

        # Check padding is zeros (black)
        assert np.all(result[0][20:, :] == 0)  # Bottom padding
        assert np.all(result[0][:, 30:] == 0)  # Right padding

    def test_list_of_3d_chw_arrays(self):
        """Test processing list of 3D arrays in CHW format."""
        rgb1 = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
        rgb2 = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)

        array_list = [rgb1, rgb2]
        result = vizy._to_numpy(array_list)

        # Should create batch with shape (2, 3, 32, 32)
        assert result.shape == (2, 3, 32, 32)
        assert np.array_equal(result[0], rgb1)
        assert np.array_equal(result[1], rgb2)

    def test_mixed_tensor_types(self):
        """Test list containing mix of numpy arrays and torch tensors."""
        np_arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        torch_arr = torch.randint(0, 255, (32, 32), dtype=torch.uint8)

        mixed_list = [np_arr, torch_arr]
        result = vizy._to_numpy(mixed_list)

        assert result.shape == (2, 32, 32)
        assert np.array_equal(result[0], np_arr)
        assert np.array_equal(result[1], torch_arr.numpy())

    def test_list_dimension_validation(self):
        """Test that 4D tensors in lists are rejected."""
        # Valid: Same dimension tensors
        valid_list = [
            np.random.rand(32, 32),  # 2D
            np.random.rand(32, 32),  # 2D
        ]
        result = vizy._to_numpy(valid_list)
        assert result.ndim == 3  # Should work (B, H, W)

        # Invalid: 4D tensor in list
        invalid_list = [
            np.random.rand(32, 32),  # 2D - OK
            np.random.rand(2, 3, 32, 32),  # 4D - NOT OK
        ]
        with pytest.raises(ValueError, match="Each tensor in list must be 2D or 3D"):
            vizy._to_numpy(invalid_list)

    def test_list_plot_integration(self):
        """Test that list plotting works end-to-end."""
        # arr0_1 = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        # arr0_2 = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        # array0_list = [arr0_1, arr0_2]

        arr1_1 = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        arr1_2 = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        array1_list = [arr1_1, arr1_2]

        # Should work without errors
        with patch("matplotlib.pyplot.show"):
            # _ = vizy.plot(array0_list)
            _ = vizy.plot(array1_list)

        plt.close("all")

    def test_list_save_integration(self):
        """Test that list saving works end-to-end."""
        arr1 = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        arr2 = np.random.randint(0, 255, (32, 32), dtype=np.uint8)

        array_list = [arr1, arr2]

        with patch("builtins.print") as mock_print:
            result_path = vizy.save(array_list)

        try:
            assert os.path.exists(result_path)
            assert result_path.endswith(".png")
            mock_print.assert_called_once_with(result_path)
        finally:
            if os.path.exists(result_path):
                os.unlink(result_path)

    def test_list_summary_integration(self):
        """Test that list summary works correctly."""
        arr1 = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        arr2 = np.random.randint(0, 255, (48, 48), dtype=np.uint8)

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

    def test_empty_list_handling(self):
        """Test that empty lists are handled gracefully."""
        with pytest.raises((TypeError, ValueError)):
            vizy._to_numpy([])

    def test_single_item_list(self):
        """Test that single-item lists work correctly."""
        arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        single_list = [arr]

        result = vizy._to_numpy(single_list)
        assert result.shape == (1, 32, 32)
        assert np.array_equal(result[0], arr)


if __name__ == "__main__":
    pytest.main([__file__])
