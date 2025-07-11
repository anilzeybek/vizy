import os

import imagehash
import numpy as np
from PIL import Image
import torch

import vizy


def get_test_image0_path() -> str:
    return "tests/data/input/test_image0.jpg"


def get_test_image0_grayscale_path() -> str:
    return "tests/data/input/test_image0_grayscale.jpg"


def get_test_image1_path() -> str:
    return "tests/data/input/test_image1.jpg"


def get_test_image2_path() -> str:
    return "tests/data/input/test_image2.jpg"


def get_test_image0() -> np.ndarray:
    return image_path_to_array(get_test_image0_path())


def get_test_image1() -> np.ndarray:
    return image_path_to_array(get_test_image1_path())


def get_test_image2() -> np.ndarray:
    return image_path_to_array(get_test_image2_path())


def image_path_to_array(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    return np.array(image)


def _resize_image_numpy(
    image: np.ndarray, height_index: int, width_index: int, target_height: int, target_width: int
) -> np.ndarray:
    """Simple nearest neighbor resize using numpy indexing."""
    h_ratio = image.shape[height_index] / target_height
    w_ratio = image.shape[width_index] / target_width
    h_indices = np.round(np.arange(target_height) * h_ratio).astype(int)
    w_indices = np.round(np.arange(target_width) * w_ratio).astype(int)
    h_indices = np.clip(h_indices, 0, image.shape[height_index] - 1)
    w_indices = np.clip(w_indices, 0, image.shape[width_index] - 1)
    resized_h = np.take(image, h_indices, axis=height_index)
    resized = np.take(resized_h, w_indices, axis=width_index)
    return resized


def images_look_same(img_path1, img_path2, tolerance=2) -> bool:
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")

    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    diff = hash1 - hash2
    return diff <= tolerance


def test_hwc():
    image = get_test_image0()
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_hw():
    image = get_test_image0()[..., 0]
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_grayscale_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_chw():
    image = get_test_image0().transpose(2, 0, 1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_1hwc():
    image = get_test_image0()[None, ...]
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_hwc1():
    image = get_test_image0()[..., None]
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_1hw():
    image = get_test_image0()[None, ..., 0]
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_grayscale_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_chw1():
    image = get_test_image0().transpose(2, 0, 1)[..., None]
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_11hw():
    image = get_test_image0()[None, None, ..., 0]
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_grayscale_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_1chw_float():
    image = get_test_image0().transpose(2, 0, 1)[None, ...] / 255.0  # Normalize to [0, 1]
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_hwc1_full_float():
    image = get_test_image0().transpose(2, 0, 1)[None, ...].astype(np.float32)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_chw1_torch():
    image = torch.from_numpy(get_test_image0().transpose(2, 0, 1)[None, ...]).float() / 255.0
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_2chw_torch():
    image0 = torch.from_numpy(get_test_image0()).permute(2, 0, 1)[None, ...]
    image1 = torch.from_numpy(get_test_image1()).permute(2, 0, 1)[None, ...]
    # Resize image1 to match image0's height and width.
    image1 = torch.nn.functional.interpolate(image1, size=(image0.shape[2], image0.shape[3]), mode="bilinear")
    image = torch.cat([image0, image1], dim=0)

    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1.png"), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_3chw():
    image0 = torch.from_numpy(get_test_image0().transpose(2, 0, 1)[None, ...])

    image1 = torch.from_numpy(get_test_image1().transpose(2, 0, 1)[None, ...])
    image1 = torch.nn.functional.interpolate(image1, size=(image0.shape[2], image0.shape[3]), mode="bilinear")

    image2 = torch.from_numpy(get_test_image2().transpose(2, 0, 1)[None, ...])
    image2 = torch.nn.functional.interpolate(image2, size=(image0.shape[2], image0.shape[3]), mode="bilinear")

    image = np.concatenate([image0, image1, image2], axis=0)  # (B, C, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-image2.png"), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_c3hw_torch():
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
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_3hwc_float():
    # Test 3 HWC images with float dtype - should match test_3chw output
    image0 = get_test_image0()[None, ...].astype(np.float32)

    image1 = get_test_image1()[None, ...].astype(np.float32)
    image1 = _resize_image_numpy(image1, 1, 2, image0.shape[1], image0.shape[2])

    image2 = get_test_image2()[None, ...].astype(np.float32)
    image2 = _resize_image_numpy(image2, 1, 2, image0.shape[1], image0.shape[2])

    image = np.concatenate([image0, image1, image2], axis=0)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-image2.png"), (
            "The saved image does not match the original."
        )
    finally:
        os.unlink(saved_image_path)


def test_2hw():
    pass


def test_4chw():
    pass


def test_4hwc():
    pass


def test_list_hwc():
    pass


def test_list_chw():
    pass


def test_list_hw():
    pass


def test_summary():
    pass


def main():
    test_hwc()


if __name__ == "__main__":
    main()
