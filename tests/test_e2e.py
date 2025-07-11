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


########################
#### 2D array tests ####
########################


def test_hw():
    image = get_test_image0()[..., 0]  # (H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_grayscale_path()), (
            "The saved image does not match the target."
        )
    finally:
        os.unlink(saved_image_path)


########################
#### 3D array tests ####
########################


def test_hwc():
    image = get_test_image0()  # (H, W, C)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        os.unlink(saved_image_path)


def test_chw():
    image = get_test_image0().transpose(2, 0, 1)  # (C, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        os.unlink(saved_image_path)


def test_1hw():
    image = get_test_image0()[None, ..., 0]  # (B=1, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_grayscale_path()), (
            "The saved image does not match the target."
        )
    finally:
        os.unlink(saved_image_path)


def test_2hw():
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
        os.unlink(saved_image_path)


########################
#### 4D array tests ####
########################


def test_1hwc():
    image = get_test_image0()[None, ...]  # (B=1, H, W, C)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        os.unlink(saved_image_path)


def test_hwc1():
    image = get_test_image0()[..., None]  # (H, W, C, B=1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        os.unlink(saved_image_path)


def test_chw1():
    image = get_test_image0().transpose(2, 0, 1)[..., None]  # (C, H, W, B=1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        os.unlink(saved_image_path)


def test_11hw():
    image = get_test_image0()[None, None, ..., 0]  # (B=1, B=1, H, W)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_grayscale_path()), (
            "The saved image does not match the target."
        )
    finally:
        os.unlink(saved_image_path)


def test_1chw_float():
    image = get_test_image0().transpose(2, 0, 1)[None, ...] / 255.0  # (C, H, W, B=1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        os.unlink(saved_image_path)


def test_hwc1_full_float():
    image = get_test_image0().transpose(2, 0, 1)[None, ...].astype(np.float32)  # (C, H, W, B=1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        os.unlink(saved_image_path)


def test_chw1_torch():
    image = torch.from_numpy(get_test_image0().transpose(2, 0, 1)[None, ...]).float() / 255.0  # (C, H, W, B=1)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, get_test_image0_path()), "The saved image does not match the target."
    finally:
        os.unlink(saved_image_path)


def test_2chw_torch():
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
        os.unlink(saved_image_path)


def test_3chw():
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
            "The saved image does not match the target."
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

    image = np.concatenate([image0, image1, image2], axis=0)  # (B=3, H, W, C)
    saved_image_path = vizy.save(image)
    try:
        assert images_look_same(saved_image_path, "tests/data/output/image0-image1-image2.png"), (
            "The saved image does not match the target."
        )
    finally:
        os.unlink(saved_image_path)


def test_4chw():
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
        os.unlink(saved_image_path)


def test_4hwc():
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
        os.unlink(saved_image_path)


########################
###### List tests ######
########################


def test_list_hwc():
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
        os.unlink(saved_image_path)


def test_list_chw():
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
        os.unlink(saved_image_path)


def test_list_hw():
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
        os.unlink(saved_image_path)


def test_summary():
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


def main():
    test_hwc()


if __name__ == "__main__":
    main()
