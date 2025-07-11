import numpy as np
import enum


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


def detect_3d_array_format(arr: np.ndarray) -> Array3DFormat:
    """Determines whether the array is in HWC, CHW, BHW, or HWB format."""
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
            else:
                return Array3DFormat.BHW  # 3 batch items
        elif channel_dim == 1:
            return Array3DFormat.HWB if d2 > d1 else Array3DFormat.BHW
        else:
            return Array3DFormat.HWC

    else:
        # Then it's either (H,W,B) or (B,H,W)
        if d0 < d1 and d0 < d2:
            return Array3DFormat.BHW
        else:
            return Array3DFormat.HWB


def _ambiguous_3d_format_detection(arr: np.ndarray) -> str:
    """
    Smart detection for ambiguous (3, H, W) tensors.
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
    rgb_score = 0  # Accumulate evidence for RGB interpretation

    # Heuristic 1: Extreme aspect ratios suggest batch of separate images
    aspect_ratio = max(h, w) / min(h, w)
    if aspect_ratio > 10:  # Very elongated suggests batch
        return "batch"

    # Heuristic 2: Check channel correlation
    # RGB images typically have moderately to highly correlated channels
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
        if correlations:
            avg_correlation = np.mean(np.abs(correlations))
            # Strong correlation suggests RGB
            if avg_correlation > 0.6:
                # But check if correlation is TOO high (might be similar noise patterns)
                if avg_correlation > 0.95:
                    # Very high correlation - might be batch with similar base patterns
                    rgb_score += 1  # Less confident
                else:
                    rgb_score += 2  # More confident
            elif avg_correlation > 0.3:
                rgb_score += 1
            # Very low correlation suggests separate images
            elif avg_correlation < 0.05:
                rgb_score -= 2
    except (np.linalg.LinAlgError, ValueError):
        pass  # Correlation failed, continue with other heuristics

    # Heuristic 3: Value range similarity
    # RGB channels often have similar value ranges
    ranges = [arr[i].max() - arr[i].min() for i in range(3)]
    mean_range = np.mean(ranges)
    if mean_range > 0:
        range_variability = np.std(ranges) / mean_range
        # Very similar ranges suggest RGB
        if range_variability < 0.2:
            rgb_score += 1
        elif range_variability < 0.4:
            rgb_score += 0.5

    # Heuristic 4: Statistical similarity across channels
    # RGB channels often have similar means and standard deviations
    means = [arr[i].mean() for i in range(3)]
    stds = [arr[i].std() for i in range(3)]

    # Check if means are reasonably similar (not too different)
    if np.mean(means) > 0:
        mean_cv = np.std(means) / np.mean(means)  # Coefficient of variation
        if mean_cv < 0.3:  # Means are quite similar
            rgb_score += 1
        elif mean_cv > 1.0:  # Means are very different
            rgb_score -= 1

    # Check if standard deviations are similar
    if np.mean(stds) > 0:
        std_cv = np.std(stds) / np.mean(stds)
        if std_cv < 0.3:  # Standard deviations are similar
            rgb_score += 0.5

    # Heuristic 5: Check for "batch-like" patterns
    # If images are very different from each other, likely a batch
    # Compare structural similarity between channels
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
            rgb_score -= 1
    except (ValueError, TypeError, np.linalg.LinAlgError):
        pass

    # Heuristic 6: Channel distinctiveness for RGB
    # RGB channels should have some distinctiveness, not be too uniform
    try:
        # Check if the channels represent different "colors" by looking at their relative intensities
        channel_maxes = [arr[i].max() for i in range(3)]
        channel_mins = [arr[i].min() for i in range(3)]

        # If all channels have very similar min/max, might be batch with similar content
        max_similarity = np.std(channel_maxes) / (np.mean(channel_maxes) + 1e-8)
        min_similarity = np.std(channel_mins) / (np.mean(channel_mins) + 1e-8)

        # RGB should have some variation in channel extremes
        if max_similarity > 0.1 or min_similarity > 0.1:
            rgb_score += 0.5
    except (ValueError, TypeError, ZeroDivisionError):
        pass

    # Decision: require strong evidence for RGB interpretation
    if rgb_score >= 2:
        return "rgb"
    else:
        return "batch"


def detect_4d_array_format(arr: np.ndarray) -> Array4DFormat:
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

    # 1) Ambiguous case – both first two dimensions look like channels (3, 3, H, W)
    if _is_channel(d0) and _is_channel(d1):
        # Only treat as ambiguous when both are exactly 3 – otherwise size 1 is
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


def _ambiguous_4d_format_detection(arr: np.ndarray) -> str:
    """
    Smart detection for ambiguous 4D tensors where both arr.shape[0] and arr.shape[1] are 3.
    Returns 'BCHW' if likely (Batch, Channel, Height, Width) or 'CBHW' if likely (Channel, Batch, Height, Width).

    Uses heuristics based on the assumption that:
    - In BCHW: each batch item should be a coherent image with correlated RGB channels
    - In CBHW: each channel should represent the same color component across all batch items

    Strong default preference for BCHW as it's the most common format in modern frameworks.
    """
    if arr.shape[0] != 3 or arr.shape[1] != 3:
        raise ValueError("This function is only for ambiguous (3, 3, H, W) arrays")

    # Start with strong BCHW preference (most common in practice)
    bchw_score = 3  # Strong default preference for BCHW
    cbhw_score = 0

    try:
        # Heuristic 1: Check correlation within putative channels vs within putative batch items

        # Interpretation 1: Assume BCHW format
        # Compare correlation within each batch item (across its 3 channels)
        bchw_correlations = []
        for b in range(3):  # For each batch item
            for c1 in range(3):
                for c2 in range(c1 + 1, 3):
                    corr = np.corrcoef(arr[b, c1].flatten(), arr[b, c2].flatten())[0, 1]
                    if not np.isnan(corr):
                        bchw_correlations.append(abs(corr))

        # Interpretation 2: Assume CBHW format
        # Compare correlation within each channel (across all batch items)
        cbhw_correlations = []
        for c in range(3):  # For each channel
            for b1 in range(3):
                for b2 in range(b1 + 1, 3):
                    corr = np.corrcoef(arr[c, b1].flatten(), arr[c, b2].flatten())[0, 1]
                    if not np.isnan(corr):
                        cbhw_correlations.append(abs(corr))

        bchw_avg_corr = np.mean(bchw_correlations) if bchw_correlations else 0
        cbhw_avg_corr = np.mean(cbhw_correlations) if cbhw_correlations else 0

        # Heuristic 2: Very high CBHW correlation with low BCHW correlation suggests CBHW
        # This would happen if we have the same image repeated 3 times with small variations
        if cbhw_avg_corr > 0.98 and bchw_avg_corr < 0.3:
            cbhw_score += 4  # Strong evidence for CBHW

        # Heuristic 3: Moderate BCHW correlation suggests natural RGB images
        if 0.2 < bchw_avg_corr < 0.95:
            bchw_score += 2  # Evidence for BCHW (natural RGB images)

        # Heuristic 4: Check the reverse pattern - in CBHW, channels of same image should correlate
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
        # this suggests CBHW format (RGB components of same image correlate better than
        # same channel across different images)
        corr_diff = cbhw_rgb_avg_corr - cbhw_avg_corr
        if corr_diff > 0.5:  # Very strong evidence
            cbhw_score += 6  # Override default BCHW preference
        elif corr_diff > 0.2:  # Strong evidence
            cbhw_score += 4  # Strong evidence for CBHW
        elif corr_diff > 0.05:  # Moderate evidence
            cbhw_score += 2  # Moderate evidence for CBHW

        # Heuristic 5: Check if each batch item looks like a coherent RGB image
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

        if rgb_like_count >= 4:  # Strong evidence of RGB-like structure
            bchw_score += 2

        # Heuristic 6: Check for CBHW-like patterns (same content, different channels)
        # Compare structural similarity across "channels" in CBHW interpretation
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

        if cbhw_similarity_score >= 6:  # Very similar content across "channels"
            cbhw_score += 2

    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        # If analysis fails, stick with default BCHW preference
        pass

    # Return result based on scores
    return "CBHW" if cbhw_score > bchw_score else "BCHW"
