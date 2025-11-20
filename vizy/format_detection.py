import enum

import numpy as np
from numpy.typing import NDArray


class Array3DFormat(enum.Enum):
    """Enum representing possible formats for 3D numpy arrays."""

    HWC = enum.auto()  # Height, Width, 3 channels
    CHW = enum.auto()  # 3 channels, Height, Width
    BHW = enum.auto()  # Batch, Height, Width
    HWB = enum.auto()  # Height, Width, Batch


class Array4DFormat(enum.Enum):
    """Enum representing possible formats for 4D numpy arrays."""

    HWCB = enum.auto()  # Height, Width, 3 channels, Batch
    CHWB = enum.auto()  # 3 channels, Height, Width, Batch
    BHWC = enum.auto()  # Batch, Height, Width, 3 channels
    BCHW = enum.auto()  # Batch, 3 channels, Height, Width
    CBHW = enum.auto()  # 3 channels, Batch, Height, Width


def detect_3d_array_format(arr: NDArray[np.number]) -> Array3DFormat:
    """Determine whether the array is in HWC, CHW, BHW, or HWB format."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.ndim}D")

    d0, d1, d2 = arr.shape
    # If any dimension is 3, it's likely a channel dimension
    # If any dimension is much larger than others, it's likely spatial (H or W)
    # Check for obvious channel dimensions
    channel_dim = None
    for i in range(3):
        if arr.shape[i] == 3:
            channel_dim = i
            break
    if channel_dim is not None:
        if channel_dim == 0:
            # Ambiguous case: (3, H, W) could be 3 channels or 3 batch items
            # Use smart_3d_format_detection to distinguish
            format_type = _ambiguous_3d_format_detection(arr)
            if format_type == "rgb":
                return Array3DFormat.CHW  # 3 color channels
            return Array3DFormat.BHW  # 3 batch items
        if channel_dim == 1:
            return Array3DFormat.HWB if d2 > d1 else Array3DFormat.BHW
        return Array3DFormat.HWC
    if d0 < d1 and d0 < d2:
        return Array3DFormat.BHW
    return Array3DFormat.HWB


def _check_aspect_ratio(h: int, w: int) -> str | None:
    """Heuristic 1: Extreme aspect ratios suggest batch of separate images."""
    aspect_ratio = max(h, w) / min(h, w)
    if aspect_ratio > 10:  # Very elongated suggests batch
        return "batch"
    return None


def _check_channel_correlation(arr: NDArray[np.number]) -> float:
    """Heuristic 2: Check channel correlation. Returns RGB score contribution."""
    try:
        # Flatten each channel and compute correlations
        ch0_flat = arr[0].flatten()
        ch1_flat = arr[1].flatten()
        ch2_flat = arr[2].flatten()

        # Compute pairwise correlations
        corr_01 = np.corrcoef(ch0_flat, ch1_flat)[0, 1]
        corr_02 = np.corrcoef(ch0_flat, ch2_flat)[0, 1]
        corr_12 = np.corrcoef(ch1_flat, ch2_flat)[0, 1]

        # Handle NaN correlations (constant channels)
        correlations = [c for c in [corr_01, corr_02, corr_12] if not np.isnan(c)]
        if not correlations:
            return 0.0

        avg_correlation = np.mean(np.abs(correlations))
        # Strong correlation suggests RGB
        if avg_correlation > 0.6:
            # But check if correlation is TOO high (might be similar noise patterns)
            if avg_correlation > 0.95:
                # Very high correlation - might be batch with similar base patterns
                return 1.0  # Less confident
            return 2.0  # More confident
        if avg_correlation > 0.3:
            return 1.0
        # Very low correlation suggests separate images
        if avg_correlation < 0.05:
            return -2.0
    except (np.linalg.LinAlgError, ValueError):
        pass  # Correlation failed, continue with other heuristics
    return 0.0


def _check_value_range_similarity(arr: NDArray[np.number]) -> float:
    """Heuristic 3: Value range similarity. Returns RGB score contribution."""
    ranges = [arr[i].max() - arr[i].min() for i in range(3)]
    mean_range = np.mean(ranges)
    if mean_range > 0:
        range_variability = np.std(ranges) / mean_range
        # Very similar ranges suggest RGB
        if range_variability < 0.2:
            return 1.0
        if range_variability < 0.4:
            return 0.5
    return 0.0


def _check_statistical_similarity(arr: NDArray[np.number]) -> float:
    """Heuristic 4: Statistical similarity across channels. Returns RGB score contribution."""
    means = [arr[i].mean() for i in range(3)]
    stds = [arr[i].std() for i in range(3)]
    score = 0.0

    # Check if means are reasonably similar (not too different)
    if np.mean(means) > 0:
        mean_cv = np.std(means) / np.mean(means)  # Coefficient of variation
        if mean_cv < 0.3:  # Means are quite similar
            score += 1.0
        elif mean_cv > 1.0:  # Means are very different
            score -= 1.0

    # Check if standard deviations are similar
    if np.mean(stds) > 0:
        std_cv = np.std(stds) / np.mean(stds)
        if std_cv < 0.3:  # Standard deviations are similar
            score += 0.5

    return score


def _check_batch_like_patterns(arr: NDArray[np.number]) -> float:
    """Heuristic 5: Check for batch-like patterns. Returns RGB score contribution."""
    try:
        # Simple structural difference: compare histograms
        hist0 = np.histogram(arr[0], bins=20, range=(arr.min(), arr.max()))[0]
        hist1 = np.histogram(arr[1], bins=20, range=(arr.min(), arr.max()))[0]
        hist2 = np.histogram(arr[2], bins=20, range=(arr.min(), arr.max()))[0]

        # Normalize histograms
        hist0 = hist0 / (np.sum(hist0) + 1e-8)
        hist1 = hist1 / (np.sum(hist1) + 1e-8)
        hist2 = hist2 / (np.sum(hist2) + 1e-8)

        # Calculate histogram differences (chi-squared like)
        diff_01 = np.sum((hist0 - hist1) ** 2)
        diff_02 = np.sum((hist0 - hist2) ** 2)
        diff_12 = np.sum((hist1 - hist2) ** 2)

        avg_hist_diff = (diff_01 + diff_02 + diff_12) / 3
        # Very different histograms suggest batch
        if avg_hist_diff > 0.1:
            return -1.0
    except (ValueError, TypeError, np.linalg.LinAlgError):
        pass
    return 0.0


def _check_channel_distinctiveness(arr: NDArray[np.number]) -> float:
    """Heuristic 6: Channel distinctiveness for RGB. Returns RGB score contribution."""
    try:
        # Check if the channels represent different "colors" by looking at their relative intensities
        channel_maxes = [arr[i].max() for i in range(3)]
        channel_mins = [arr[i].min() for i in range(3)]

        # If all channels have very similar min/max, might be batch with similar content
        max_similarity = np.std(channel_maxes) / (np.mean(channel_maxes) + 1e-8)
        min_similarity = np.std(channel_mins) / (np.mean(channel_mins) + 1e-8)

        # RGB should have some variation in channel extremes
        if max_similarity > 0.1 or min_similarity > 0.1:
            return 0.5
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return 0.0


def _ambiguous_3d_format_detection(arr: NDArray[np.number]) -> str:
    """Smart detection for ambiguous (3, H, W) tensors.

    Returns 'rgb' if likely RGB image, 'batch' if likely 3 grayscale images.

    Uses multiple heuristics:
    1. Aspect ratio - very wide/tall suggests batch of images
    2. Channel correlation - RGB channels are usually more correlated
    3. Value range similarity across channels
    4. Statistical similarity across channels
    5. Pattern distinctiveness - very similar patterns suggest batch with variations

    Conservative approach: defaults to 'batch' unless strong evidence for RGB.
    """
    if arr.shape[0] != 3:
        raise ValueError("This function is only for (3, H, W) arrays")

    h, w = arr.shape[1], arr.shape[2]

    # Heuristic 1: Extreme aspect ratios suggest batch of separate images
    aspect_result = _check_aspect_ratio(h, w)
    if aspect_result:
        return aspect_result

    # Accumulate evidence for RGB interpretation
    rgb_score = (
        _check_channel_correlation(arr)
        + _check_value_range_similarity(arr)
        + _check_statistical_similarity(arr)
        + _check_batch_like_patterns(arr)
        + _check_channel_distinctiveness(arr)
    )

    # Decision: require strong evidence for RGB interpretation
    return "rgb" if rgb_score >= 2 else "batch"


def detect_4d_array_format(arr: NDArray[np.number]) -> Array4DFormat:
    """Determine the format of a 4-D numpy array.

    Supported layouts (where B - batch, C - channel, H - height, W - width):

    1. BHWC  - (B, H, W, C)
    2. BCHW  - (B, C, H, W)
    3. CBHW - (C, B, H, W)
    4. CHWB - (C, H, W, B)
    5. HWCB  - (H, W, C, B)

    The function treats a dimension of size **3** (or **1** for grayscale) as a strong
    indicator of the *channel* axis.  When both the first two axes have size 3 the
    layout is ambiguous - we fall back to the heuristics implemented in
    ``_ambiguous_4d_format_detection``.
    """
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array, got {arr.ndim}D")

    d0, d1, d2, d3 = arr.shape

    # Helper to check if a dimension could reasonably be the channel axis
    def _is_channel(dim_size: int) -> bool:
        return dim_size in (1, 3)

    # 1) Ambiguous case: both first two dimensions look like channels (3, 3, H, W)
    if _is_channel(d0) and _is_channel(d1):
        # Only treat as ambiguous when both are exactly 3: otherwise size 1 is
        # more likely to be a singleton batch or channel and easy to disambiguate.
        if d0 == 3 and d1 == 3:
            interpretation = _ambiguous_4d_format_detection(arr)
            return Array4DFormat.BCHW if interpretation == "BCHW" else Array4DFormat.CBHW
        # If one (or both) of them is 1 we can assume axis-0 is channel and axis-1
        # is batch because height/width rarely equal 1.
        return Array4DFormat.CBHW

    # 2) Clear channel axis based on where the 3/1 is located
    if _is_channel(d3):  # (B, H, W, C)
        return Array4DFormat.BHWC
    if _is_channel(d2):  # (H, W, C, B)
        return Array4DFormat.HWCB
    if _is_channel(d1):  # (B, C, H, W)
        return Array4DFormat.BCHW
    if _is_channel(d0):  # Either (C, B, H, W) or (C, H, W, B)
        # Heuristic: whichever of dims 1 or 3 is smaller is probably the batch axis
        # (batch size is usually smaller than spatial dimensions).
        return Array4DFormat.CBHW if d1 <= d3 else Array4DFormat.CHWB

    # 3) If no dimension looks like a channel we cannot determine the format
    raise ValueError(f"Unable to determine 4D array format for shape {arr.shape}")


def _compute_bchw_correlations(arr: NDArray[np.number]) -> list[float]:
    """Compute correlations within each batch item (across its 3 channels) for BCHW interpretation."""
    bchw_correlations = []
    for b in range(3):  # For each batch item
        for c1 in range(3):
            for c2 in range(c1 + 1, 3):
                corr = np.corrcoef(arr[b, c1].flatten(), arr[b, c2].flatten())[0, 1]
                if not np.isnan(corr):
                    bchw_correlations.append(abs(corr))
    return bchw_correlations


def _compute_cbhw_correlations(arr: NDArray[np.number]) -> list[float]:
    """Compute correlations within each channel (across all batch items) for CBHW interpretation."""
    cbhw_correlations = []
    for c in range(3):  # For each channel
        for b1 in range(3):
            for b2 in range(b1 + 1, 3):
                corr = np.corrcoef(arr[c, b1].flatten(), arr[c, b2].flatten())[0, 1]
                if not np.isnan(corr):
                    cbhw_correlations.append(abs(corr))
    return cbhw_correlations


def _check_correlation_heuristics(bchw_avg_corr: float, cbhw_avg_corr: float) -> tuple[float, float]:
    """Heuristics 2-3: Check correlation patterns. Returns (bchw_score, cbhw_score)."""
    bchw_score = 0.0
    cbhw_score = 0.0

    # Heuristic 2: Very high CBHW correlation with low BCHW correlation suggests CBHW
    if cbhw_avg_corr > 0.98 and bchw_avg_corr < 0.3:
        cbhw_score += 4  # Strong evidence for CBHW

    # Heuristic 3: Moderate BCHW correlation suggests natural RGB images
    if 0.2 < bchw_avg_corr < 0.95:
        bchw_score += 2  # Evidence for BCHW (natural RGB images)

    return bchw_score, cbhw_score


def _check_reverse_correlation_pattern(arr: NDArray[np.number], cbhw_avg_corr: float) -> float:
    """Heuristic 4: Check reverse pattern - CBHW RGB correlation vs within-channel correlation."""
    # For each "batch position" in CBHW interpretation, check RGB correlation
    cbhw_rgb_corrs = []
    for b in range(3):  # For each batch position in CBHW interpretation
        for c1 in range(3):
            for c2 in range(c1 + 1, 3):
                corr = np.corrcoef(arr[c1, b].flatten(), arr[c2, b].flatten())[0, 1]
                if not np.isnan(corr):
                    cbhw_rgb_corrs.append(abs(corr))

    cbhw_rgb_avg_corr = np.mean(cbhw_rgb_corrs) if cbhw_rgb_corrs else 0

    # If CBHW RGB correlation is significantly higher than within-channel correlation,
    # this suggests CBHW format
    corr_diff = cbhw_rgb_avg_corr - cbhw_avg_corr
    if corr_diff > 0.5:  # Very strong evidence
        return 6.0  # Override default BCHW preference
    if corr_diff > 0.2:  # Strong evidence
        return 4.0  # Strong evidence for CBHW
    if corr_diff > 0.05:  # Moderate evidence
        return 2.0  # Moderate evidence for CBHW
    return 0.0


def _check_rgb_like_structure(arr: NDArray[np.number]) -> float:
    """Heuristic 5: Check if each batch item looks like a coherent RGB image."""
    rgb_like_count = 0
    for b in range(3):
        # Statistical diversity check: RGB channels should have different characteristics
        means = [arr[b, c].mean() for c in range(3)]
        stds = [arr[b, c].std() for c in range(3)]

        # RGB channels often have different means and similar stds
        if len(set(np.round(means, 1))) > 1:  # Different means
            rgb_like_count += 1
        if np.std(stds) / (np.mean(stds) + 1e-8) < 0.5:  # Similar standard deviations
            rgb_like_count += 1

    # Strong evidence of RGB-like structure
    return 2.0 if rgb_like_count >= 4 else 0.0


def _check_cbhw_similarity_pattern(arr: NDArray[np.number]) -> float:
    """Heuristic 6: Check for CBHW-like patterns using histogram similarity."""
    cbhw_similarity_score = 0
    for c in range(3):
        for b1 in range(3):
            for b2 in range(b1 + 1, 3):
                # Simple structural similarity: compare histograms
                hist1 = np.histogram(arr[c, b1], bins=10, range=(arr.min(), arr.max()))[0]
                hist2 = np.histogram(arr[c, b2], bins=10, range=(arr.min(), arr.max()))[0]
                hist1 = hist1 / (np.sum(hist1) + 1e-8)
                hist2 = hist2 / (np.sum(hist2) + 1e-8)

                # High histogram similarity suggests same content (CBHW pattern)
                similarity = 1 - np.sum(np.abs(hist1 - hist2)) / 2
                if similarity > 0.8:
                    cbhw_similarity_score += 1

    # Very similar content across "channels"
    return 2.0 if cbhw_similarity_score >= 6 else 0.0


def _ambiguous_4d_format_detection(arr: NDArray[np.number]) -> str:
    """Smart detection for ambiguous 4D tensors where both arr.shape[0] and arr.shape[1] are 3.

    Returns 'BCHW' if likely (Batch, Channel, Height, Width) or 'CBHW' if likely (Channel, Batch, Height, Width).

    Uses heuristics based on the assumption that:
    - In BCHW: each batch item should be a coherent image with correlated RGB channels
    - In CBHW: each channel should represent the same color component across all batch items

    Strong default preference for BCHW as it's the most common format in modern frameworks.
    """
    if arr.shape[0] != 3 or arr.shape[1] != 3:
        raise ValueError("This function is only for ambiguous (3, 3, H, W) arrays")

    # Start with strong BCHW preference (most common in practice)
    bchw_score = 3.0  # Strong default preference for BCHW
    cbhw_score = 0.0

    try:
        # Heuristic 1: Check correlation within putative channels vs within putative batch items
        bchw_correlations = _compute_bchw_correlations(arr)
        cbhw_correlations = _compute_cbhw_correlations(arr)

        bchw_avg_corr = float(np.mean(bchw_correlations)) if bchw_correlations else 0.0
        cbhw_avg_corr = float(np.mean(cbhw_correlations)) if cbhw_correlations else 0.0

        # Heuristics 2-3: Check correlation patterns
        bchw_add, cbhw_add = _check_correlation_heuristics(bchw_avg_corr, cbhw_avg_corr)
        bchw_score += bchw_add
        cbhw_score += cbhw_add

        # Heuristic 4: Check reverse correlation pattern
        cbhw_score += _check_reverse_correlation_pattern(arr, cbhw_avg_corr)

        # Heuristic 5: Check if each batch item looks like a coherent RGB image
        bchw_score += _check_rgb_like_structure(arr)

        # Heuristic 6: Check for CBHW-like patterns
        cbhw_score += _check_cbhw_similarity_pattern(arr)

    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        # If analysis fails, stick with default BCHW preference
        pass

    # Return result based on scores
    return "CBHW" if cbhw_score > bchw_score else "BCHW"
