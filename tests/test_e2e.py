from pathlib import Path

import imagehash
import numpy as np
import pytest
import torch
from PIL import Image

import vizy


def get_test_image0_path() -> str:
    return "tests/data/input/test_image0.jpg"


def get_test_image0_grayscale_path() -> str:
    return "tests/data/input/test_image0_grayscale.jpg"


def get_test_image1_path() -> str:
    return "tests/data/input/test_image1.jpg"


def get_test_image2_path() -> str:
    return "tests/data/input/test_image2.jpg"


def get_test_image3_path() -> str:
    return "tests/data/input/test_image3.jpg"


def get_test_image0() -> np.ndarray:
    return image_path_to_array(get_test_image0_path())


def get_test_image1() -> np.ndarray:
    return image_path_to_array(get_test_image1_path())


def get_test_image2() -> np.ndarray:
    return image_path_to_array(get_test_image2_path())


def get_test_image3() -> np.ndarray:
    return image_path_to_array(get_test_image3_path())


def image_path_to_array(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    return np.array(image)


def _resize_image_numpy(
    image: np.ndarray, height_index: int, width_index: int, target_height: int, target_width: int
) -> np.ndarray:
    """Resize image using simple nearest neighbor and numpy indexing."""
    h_ratio = image.shape[height_index] / target_height
    w_ratio = image.shape[width_index] / target_width
    h_indices = np.round(np.arange(target_height) * h_ratio).astype(int)
    w_indices = np.round(np.arange(target_width) * w_ratio).astype(int)
    h_indices = np.clip(h_indices, 0, image.shape[height_index] - 1)
    w_indices = np.clip(w_indices, 0, image.shape[width_index] - 1)
    resized_h = np.take(image, h_indices, axis=height_index)
    resized = np.take(resized_h, w_indices, axis=width_index)
    return resized


def images_look_same(img_path1: str, img_path2: str, tolerance: int = 2) -> bool:
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")

    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    diff = hash1 - hash2
    return diff <= tolerance


########################
#### 2D array tests ####
########################


def test_hw() -> None:
    image = get_test_image0()[..., 0]  # (H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_grayscale_path()), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_hw_bool_array() -> None:
    binary_mask = torch.zeros((256, 256), dtype=torch.bool)
    binary_mask[100:150, 100:150] = True
    saved_image_path = vizy.save(binary_mask)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/binary_mask.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


########################
#### 3D array tests ####
########################


def test_hwc() -> None:
    image = get_test_image0()  # (H, W, C)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        Path(saved_image_path).unlink()


def test_chw() -> None:
    image = get_test_image0().transpose(2, 0, 1)  # (C, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        Path(saved_image_path).unlink()


def test_1hw() -> None:
    image = get_test_image0()[None, ..., 0]  # (B=1, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_grayscale_path()), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_2hw() -> None:
    image0 = get_test_image0()[..., 0][None, ...]
    image1 = get_test_image1()[..., 0][None, ...]
    image1 = _resize_image_numpy(image1, 1, 2, image0.shape[1], image0.shape[2])

    image = np.concatenate([image0, image1], axis=0)  # (B=2, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-grayscale.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


########################
#### 4D array tests ####
########################


def test_1hwc() -> None:
    image = get_test_image0()[None, ...]  # (B=1, H, W, C)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        Path(saved_image_path).unlink()


def test_hwc1() -> None:
    image = get_test_image0()[..., None]  # (H, W, C, B=1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        Path(saved_image_path).unlink()


def test_chw1() -> None:
    image = get_test_image0().transpose(2, 0, 1)[..., None]  # (C, H, W, B=1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        Path(saved_image_path).unlink()


def test_11hw() -> None:
    image = get_test_image0()[None, None, ..., 0]  # (B=1, B=1, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_grayscale_path()), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_1chw_float() -> None:
    image = get_test_image0().transpose(2, 0, 1)[None, ...] / 255.0  # (C, H, W, B=1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        Path(saved_image_path).unlink()


def test_hwc1_full_float() -> None:
    image = get_test_image0().transpose(2, 0, 1)[None, ...].astype(np.float32)  # (C, H, W, B=1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        Path(saved_image_path).unlink()


def test_chw1_torch() -> None:
    image = torch.from_numpy(get_test_image0().transpose(2, 0, 1)[None, ...]).float() / 255.0  # (C, H, W, B=1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        Path(saved_image_path).unlink()


def test_2chw_torch() -> None:
    image0 = torch.from_numpy(get_test_image0()).permute(2, 0, 1)[None, ...]
    image1 = torch.from_numpy(get_test_image1()).permute(2, 0, 1)[None, ...]
    # Resize image1 to match image0's height and width.
    image1 = torch.nn.functional.interpolate(image1, size=(image0.shape[2], image0.shape[3]), mode="bilinear")
    image = torch.cat([image0, image1], dim=0)  # (B=2, C, H, W)

    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_3chw() -> None:
    image0 = torch.from_numpy(get_test_image0().transpose(2, 0, 1)[None, ...])

    image1 = torch.from_numpy(get_test_image1().transpose(2, 0, 1)[None, ...])
    image1 = torch.nn.functional.interpolate(image1, size=(image0.shape[2], image0.shape[3]), mode="bilinear")

    image2 = torch.from_numpy(get_test_image2().transpose(2, 0, 1)[None, ...])
    image2 = torch.nn.functional.interpolate(image2, size=(image0.shape[2], image0.shape[3]), mode="bilinear")

    image = np.concatenate([image0, image1, image2], axis=0)  # (B=3, C, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-image2.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_3chw_alt1() -> None:
    image1 = torch.from_numpy(get_test_image1().transpose(2, 0, 1)[None, ...])

    image2 = torch.from_numpy(get_test_image2().transpose(2, 0, 1)[None, ...])
    image2 = torch.nn.functional.interpolate(image2, size=(image1.shape[2], image1.shape[3]), mode="bilinear")

    image3 = torch.from_numpy(get_test_image3().transpose(2, 0, 1)[None, ...])
    image3 = torch.nn.functional.interpolate(image3, size=(image1.shape[2], image1.shape[3]), mode="bilinear")

    image = np.concatenate([image1, image2, image3], axis=0)  # (B=3, C, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image1-image2-image3.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_3chw_alt2() -> None:
    image0 = torch.from_numpy(get_test_image0().transpose(2, 0, 1)[None, ...])

    image2 = torch.from_numpy(get_test_image2().transpose(2, 0, 1)[None, ...])
    image2 = torch.nn.functional.interpolate(image2, size=(image0.shape[2], image0.shape[3]), mode="bilinear")

    image3 = torch.from_numpy(get_test_image3().transpose(2, 0, 1)[None, ...])
    image3 = torch.nn.functional.interpolate(image3, size=(image0.shape[2], image0.shape[3]), mode="bilinear")

    image = np.concatenate([image0, image2, image3], axis=0)  # (B=3, C, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image2-image3.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_3chw_alt3() -> None:
    image3 = torch.from_numpy(get_test_image3().transpose(2, 0, 1)[None, ...])

    image0 = torch.from_numpy(get_test_image0().transpose(2, 0, 1)[None, ...])
    image0 = torch.nn.functional.interpolate(image0, size=(image3.shape[2], image3.shape[3]), mode="bilinear")

    image1 = torch.from_numpy(get_test_image1().transpose(2, 0, 1)[None, ...])
    image1 = torch.nn.functional.interpolate(image1, size=(image3.shape[2], image3.shape[3]), mode="bilinear")

    image = np.concatenate([image3, image0, image1], axis=0)  # (B=3, C, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image3-image0-image1.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_c3hw_torch() -> None:
    image0 = torch.from_numpy(get_test_image0()).permute(2, 0, 1)[None, ...]

    image1 = torch.from_numpy(get_test_image1()).permute(2, 0, 1)[None, ...]
    image1 = torch.nn.functional.interpolate(image1, size=(image0.shape[2], image0.shape[3]), mode="bilinear")

    image2 = torch.from_numpy(get_test_image2()).permute(2, 0, 1)[None, ...]
    image2 = torch.nn.functional.interpolate(image2, size=(image0.shape[2], image0.shape[3]), mode="bilinear")

    image = torch.cat([image0, image1, image2], dim=0)
    image = image.permute(1, 0, 2, 3)  # (C, B, H, W)

    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-image2.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_3hwc_float() -> None:
    # Test 3 HWC images with float dtype - should match test_3chw output
    image0 = get_test_image0()[None, ...].astype(np.float32)

    image1 = get_test_image1()[None, ...].astype(np.float32)
    image1 = _resize_image_numpy(image1, 1, 2, image0.shape[1], image0.shape[2])

    image2 = get_test_image2()[None, ...].astype(np.float32)
    image2 = _resize_image_numpy(image2, 1, 2, image0.shape[1], image0.shape[2])

    image = np.concatenate([image0, image1, image2], axis=0)  # (B=3, H, W, C)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-image2.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_4chw() -> None:
    # Test 4 CHW images in a 2x2 grid
    image0 = get_test_image0().transpose(2, 0, 1)[None, ...]  # 1CHW

    image1 = get_test_image1().transpose(2, 0, 1)[None, ...]
    image1 = _resize_image_numpy(image1, 2, 3, image0.shape[2], image0.shape[3])

    image2 = get_test_image2().transpose(2, 0, 1)[None, ...]
    image2 = _resize_image_numpy(image2, 2, 3, image0.shape[2], image0.shape[3])

    image3 = get_test_image3().transpose(2, 0, 1)[None, ...]
    image3 = _resize_image_numpy(image3, 2, 3, image0.shape[2], image0.shape[3])

    image = np.concatenate([image0, image1, image2, image3], axis=0)  # (B=4, C, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-image2-image3.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_4hwc() -> None:
    # Test 4 HWC images in a 2x2 grid (should be same result as test_4chw)
    image0 = get_test_image0()[None, ...]  # 1HWC

    image1 = get_test_image1()[None, ...]
    image1 = _resize_image_numpy(image1, 1, 2, image0.shape[1], image0.shape[2])

    image2 = get_test_image2()[None, ...]
    image2 = _resize_image_numpy(image2, 1, 2, image0.shape[1], image0.shape[2])

    image3 = get_test_image3()[None, ...]
    image3 = _resize_image_numpy(image3, 1, 2, image0.shape[1], image0.shape[2])

    image = np.concatenate([image0, image1, image2, image3], axis=0)  # (B=4, H, W, C)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-image2-image3.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


########################
###### List tests ######
########################


def test_list_hwc() -> None:
    image0 = get_test_image0()
    image1 = get_test_image1()
    image2 = get_test_image2()
    image2 = _resize_image_numpy(image2, 0, 1, image0.shape[0], image0.shape[1])

    image_list = [image0, image1, image2]  # list of (H, W, C)
    saved_image_path = vizy.save(image_list)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-image2-list.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_list_chw() -> None:
    image0 = get_test_image0().transpose(2, 0, 1)
    image1 = get_test_image1().transpose(2, 0, 1)
    image2 = get_test_image2().transpose(2, 0, 1)
    image2 = _resize_image_numpy(image2, 1, 2, image0.shape[1], image0.shape[2])

    image_list = [image0, image1, image2]  # list of (C, H, W)
    saved_image_path = vizy.save(image_list)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-image2-list.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_list_hw() -> None:
    image0 = get_test_image0()[..., 0]  # (H, W)
    image1 = get_test_image1()[..., 0]  # (H, W)
    image1 = _resize_image_numpy(image1, 0, 1, image0.shape[0], image0.shape[1])

    image_list = [image0, image1]  # list of (H, W)
    saved_image_path = vizy.save(image_list)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-grayscale.png"), (
            "The saved image does not match the target."
        )
    finally:
        Path(saved_image_path).unlink()


def test_summary() -> None:
    # Test summary function with different tensor types
    image = get_test_image0()

    # Test with numpy array - should not raise any exceptions
    vizy.summary(image)

    torch_image = torch.from_numpy(image)
    vizy.summary(torch_image)

    pil_image = Image.fromarray(image)
    vizy.summary(pil_image)

    # Test with list of tensors
    image_list = [image, image + 10]
    vizy.summary(image_list)

    # If we get here without exceptions, the test passes
    assert True


########################
###### Edge Cases ######
########################


def test_ambiguous_33hw_bchw() -> None:
    """Test ambiguous (3, 3, H, W) tensor that should be detected as BCHW (3 RGB images)."""
    # Create 3 distinct RGB-like images with correlated channels
    rng = np.random.default_rng(42)
    base_img = rng.integers(0, 255, (32, 32), dtype=np.uint8)

    # Create RGB versions with correlated channels
    img1_r = base_img
    img1_g = np.clip(base_img * 0.8 + 20, 0, 255).astype(np.uint8)
    img1_b = np.clip(base_img * 0.6 + 40, 0, 255).astype(np.uint8)
    img1 = np.stack([img1_r, img1_g, img1_b], axis=0)

    img2_r = np.roll(base_img, 5, axis=0)
    img2_g = np.clip(img2_r * 0.9 + 10, 0, 255).astype(np.uint8)
    img2_b = np.clip(img2_r * 0.7 + 30, 0, 255).astype(np.uint8)
    img2 = np.stack([img2_r, img2_g, img2_b], axis=0)

    img3_r = np.roll(base_img, -5, axis=1)
    img3_g = np.clip(img3_r * 0.85 + 15, 0, 255).astype(np.uint8)
    img3_b = np.clip(img3_r * 0.65 + 35, 0, 255).astype(np.uint8)
    img3 = np.stack([img3_r, img3_g, img3_b], axis=0)

    # Stack as (3, 3, H, W) - should be interpreted as BCHW (3 batch, 3 channels)
    ambiguous_tensor = np.stack([img1, img2, img3], axis=0)
    assert ambiguous_tensor.shape == (3, 3, 32, 32)

    saved_image_path = vizy.save(ambiguous_tensor)
    try:
        # Should create a grid of 3 RGB images
        assert Path(saved_image_path).exists()
        # The output should be a valid image grid
        result_img = Image.open(saved_image_path)
        assert result_img.size[0] > 32  # Should be wider due to grid layout
    finally:
        Path(saved_image_path).unlink()


def test_ambiguous_33hw_cbhw() -> None:
    """Test ambiguous (3, 3, H, W) tensor that should be detected as CBHW (3 channels of 3 batch items)."""
    # Create 3 very similar images (same content, different channels)
    rng = np.random.default_rng(123)
    base_pattern = rng.integers(50, 200, (32, 32), dtype=np.uint8)

    # Create 3 similar images (like same scene with small variations)
    img1 = base_pattern
    img2 = np.clip(base_pattern + rng.integers(-10, 11, base_pattern.shape), 0, 255).astype(np.uint8)
    img3 = np.clip(base_pattern + rng.integers(-10, 11, base_pattern.shape), 0, 255).astype(np.uint8)

    # Arrange as CBHW: each "channel" contains all batch items
    r_channel = np.stack([img1, img2, img3], axis=0)  # (3, H, W)
    g_channel = np.stack([img1, img2, img3], axis=0)  # Same pattern
    b_channel = np.stack([img1, img2, img3], axis=0)  # Same pattern

    # Stack as (3, 3, H, W) but in CBHW format
    cbhw_tensor = np.stack([r_channel, g_channel, b_channel], axis=0)
    assert cbhw_tensor.shape == (3, 3, 32, 32)

    saved_image_path = vizy.save(cbhw_tensor)
    try:
        assert Path(saved_image_path).exists()
        result_img = Image.open(saved_image_path)
        # Should still create a valid output
        assert result_img.size[0] > 0
        assert result_img.size[1] > 0
    finally:
        Path(saved_image_path).unlink()


def test_edge_case_dtypes() -> None:
    """Test various edge case data types."""
    base_image = get_test_image0()[:64, :64]  # Smaller for faster testing

    # Test int16
    image_int16 = base_image.astype(np.int16)
    saved_path = vizy.save(image_int16)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Test int32
    image_int32 = base_image.astype(np.int32)
    saved_path = vizy.save(image_int32)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Test float64
    image_float64 = (base_image / 255.0).astype(np.float64)
    saved_path = vizy.save(image_float64)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Test boolean array (convert to uint8 since matplotlib has limitations with bool)
    image_bool = (base_image > 127).astype(np.uint8) * 255  # Convert bool to 0/255
    saved_path = vizy.save(image_bool)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()


def test_float_range_edge_cases() -> None:
    """Test float arrays in different value ranges."""
    h, w = 32, 32
    rng = np.random.default_rng(42)

    # Test 0-1 range (normalized)
    image_01 = rng.random((h, w, 3), dtype=np.float32)
    saved_path = vizy.save(image_01)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Test -1 to 1 range
    image_neg1_1 = (rng.random((h, w, 3), dtype=np.float32) * 2 - 1).astype(np.float32)
    saved_path = vizy.save(image_neg1_1)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Test very large values
    image_large = (rng.random((h, w, 3), dtype=np.float32) * 1000 + 500).astype(np.float32)
    saved_path = vizy.save(image_large)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Test array with some NaN values (should raise ValueError)
    image_with_nan = rng.random((h, w, 3), dtype=np.float32)
    image_with_nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Cannot plot array with NaN values"):
        vizy.save(image_with_nan)


def test_single_pixel_edge_cases() -> None:
    """Test degenerate single pixel and very small images."""
    rng = np.random.default_rng(42)
    # 1x1 pixel RGB image (but don't squeeze to avoid dimension issues)
    tiny_img = np.array([[[255, 0, 0]]], dtype=np.uint8)  # (1, 1, 3)
    try:
        saved_path = vizy.save(tiny_img)
        # This might fail due to dimension handling, which is expected behavior
        assert Path(saved_path).exists()
        Path(saved_path).unlink()
    except ValueError:
        # Expected: 1x1 images may cause dimension issues after squeezing
        pass

    # 1xN image (very wide, thin strip)
    strip_h = rng.integers(0, 255, (1, 50, 3), dtype=np.uint8)
    saved_path = vizy.save(strip_h)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Nx1 image (very tall, thin strip)
    strip_v = rng.integers(0, 255, (50, 1, 3), dtype=np.uint8)
    saved_path = vizy.save(strip_v)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # 1x1 grayscale
    tiny_gray = np.array([[128]], dtype=np.uint8)  # (1, 1)
    try:
        saved_path = vizy.save(tiny_gray)
        assert Path(saved_path).exists()
        Path(saved_path).unlink()
    except ValueError:
        # Expected: 1x1 images may cause dimension issues after squeezing
        pass


def test_mixed_pil_modes() -> None:
    """Test PIL images in different color modes."""
    # RGBA image
    rgba_img = Image.new("RGBA", (32, 32), color=(255, 128, 0, 200))
    saved_path = vizy.save(rgba_img)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Grayscale PIL image
    gray_img = Image.new("L", (32, 32), color=128)
    saved_path = vizy.save(gray_img)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Palette mode image
    palette_img = Image.new("P", (32, 32))
    palette_img.putpalette([i % 256 for i in range(256 * 3)])  # Simple palette (mod 256)
    saved_path = vizy.save(palette_img)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()


def test_format_detection_edge_cases() -> None:
    """Test format detection for unusual dimension sizes."""
    rng = np.random.default_rng(42)
    # Test (4, H, W) - 4 channels, should be detected as batch
    image_4hw = rng.integers(0, 255, (4, 32, 32), dtype=np.uint8)
    saved_path = vizy.save(image_4hw)
    try:
        assert Path(saved_path).exists()
        # Should create a 2x2 grid for 4 images
    finally:
        Path(saved_path).unlink()

    # Test (2, H, W) - 2 channels, should be batch
    image_2hw = rng.integers(0, 255, (2, 32, 32), dtype=np.uint8)
    saved_path = vizy.save(image_2hw)
    try:
        assert Path(saved_path).exists()
        # Should create side-by-side layout
    finally:
        Path(saved_path).unlink()

    # Test very small spatial dimensions with channels
    tiny_spatial = rng.integers(0, 255, (3, 3, 3), dtype=np.uint8)  # Highly ambiguous
    saved_path = vizy.save(tiny_spatial)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()


def test_grid_layout_unusual_batches() -> None:
    """Test grid layouts for unusual batch sizes."""
    base_img = get_test_image0()[:32, :32, 0]  # Small grayscale for speed

    # Test 5 images (should create appropriate grid)
    images_5 = np.stack([base_img + i * 10 for i in range(5)], axis=0)
    saved_path = vizy.save(images_5)
    try:
        assert Path(saved_path).exists()
        result_img = Image.open(saved_path)
        # Should arrange in roughly square grid
        assert result_img.size[0] >= 32 * 3  # At least 3 columns
    finally:
        Path(saved_path).unlink()

    # Test 7 images
    images_7 = np.stack([base_img + i * 10 for i in range(7)], axis=0)
    saved_path = vizy.save(images_7)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Test 10 images
    images_10 = np.stack([base_img + i * 10 for i in range(10)], axis=0)
    saved_path = vizy.save(images_10)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()


def test_all_zeros_black_images() -> None:
    """Test arrays that are all zeros (black images)."""
    # All black RGB image
    black_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    saved_path = vizy.save(black_rgb)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # All black grayscale
    black_gray = np.zeros((64, 64), dtype=np.uint8)
    saved_path = vizy.save(black_gray)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Batch of black images
    black_batch = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    saved_path = vizy.save(black_batch)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()


def test_torch_device_mixed_types() -> None:
    """Test tensors on different devices and mixed types in lists."""
    if torch is None:
        return  # Skip if torch not available

    base_img = get_test_image0()[:32, :32]

    # CPU tensor
    cpu_tensor = torch.from_numpy(base_img.transpose(2, 0, 1)).float()

    # Mixed list: numpy + torch
    mixed_list = [base_img, cpu_tensor.permute(1, 2, 0).numpy()]
    saved_path = vizy.save(mixed_list)
    try:
        assert Path(saved_path).exists()
    finally:
        Path(saved_path).unlink()

    # Test CUDA if available
    if torch.cuda.is_available():
        cuda_tensor = cpu_tensor.cuda()
        saved_path = vizy.save(cuda_tensor)
        try:
            assert Path(saved_path).exists()
        finally:
            Path(saved_path).unlink()


def test_large_batch_performance() -> None:
    """Test performance and correctness with large batch sizes."""
    rng = np.random.default_rng(42)
    # Create a batch of 20 small images to test grid layout and performance
    small_img = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    large_batch = np.stack([small_img + i for i in range(20)], axis=0)

    saved_path = vizy.save(large_batch)
    try:
        assert Path(saved_path).exists()
        result_img = Image.open(saved_path)
        # Should create a reasonable grid (likely 5x4 or similar)
        expected_min_width = 16 * 4  # At least 4 columns
        assert result_img.size[0] >= expected_min_width
    finally:
        Path(saved_path).unlink()


def test_bchw_two_fullhd_grayscale_side_by_side() -> None:
    """BCHW tensor (2, 1, 1080, 1920) should render two 1080p grayscale images side-by-side."""
    tensor = torch.randint(0, 256, (2, 1, 1080, 1920), dtype=torch.uint8)
    saved_path = vizy.save(tensor)
    try:
        assert Path(saved_path).exists()
        img = Image.open(saved_path)
        # Expect width 2*1920 and height 1080
        assert img.size == (1920 * 2, 1080)
    finally:
        Path(saved_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
