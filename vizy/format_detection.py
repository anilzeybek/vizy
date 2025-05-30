import numpy as np


def smart_3d_format_detection(arr: np.ndarray) -> str:
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


def smart_4d_format_detection(arr: np.ndarray) -> str:
    """
    Smart detection for ambiguous 4D tensors where both arr.shape[0] and arr.shape[1] are 3.
    Returns 'BCHW' if likely (Batch, Channel, Height, Width) or 'CBHW' if likely (Channel, Batch, Height, Width).

    Uses heuristics based on the assumption that:
    - In BCHW: each batch item should be a coherent image
    - In CBHW: each channel should represent the same color component across all batch items
    """
    if arr.shape[0] != 3 or arr.shape[1] != 3:
        raise ValueError("This function is only for ambiguous (3, 3, H, W) arrays")

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

    # RGB channels typically have moderate correlation, while batch items of the same channel
    # (especially with noise variations) should have high correlation
    bchw_avg_corr = np.mean(bchw_correlations) if bchw_correlations else 0
    cbhw_avg_corr = np.mean(cbhw_correlations) if cbhw_correlations else 0

    # Decision logic:
    # If CBHW interpretation shows high correlation (same channel across batch items with variations)
    # and BCHW shows lower correlation, prefer CBHW
    cbhw_score = 0
    bchw_score = 0

    # High correlation within same channel across batch items suggests CBHW
    if cbhw_avg_corr > bchw_avg_corr + 0.1:  # Significantly higher
        cbhw_score += 2
    elif cbhw_avg_corr > 0.8:  # Very high correlation suggests same channel
        cbhw_score += 2
    elif cbhw_avg_corr > 0.6:
        cbhw_score += 1

    # Moderate correlation within RGB channels suggests BCHW
    if bchw_avg_corr > 0.3 and bchw_avg_corr < 0.8:
        bchw_score += 1

    # Default preference: BCHW is more common in most frameworks
    bchw_score += 0.5

    return "CBHW" if cbhw_score > bchw_score else "BCHW"
