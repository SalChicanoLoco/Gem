"""
Image similarity metrics (agents/image_metrics.py)

Real measurements comparing two images. Every value returned here is computed
from pixel data; nothing is estimated from iteration counts or step counts.

Provides:
  - ``ssim``: global structural similarity on grayscale, in [-1, 1].
  - ``histogram_correlation``: per-channel colour histogram correlation, in [-1, 1].
  - ``compare``: both of the above plus a combined 0-100 score.
"""

import logging
from typing import Any, Dict

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Standard SSIM stabilisers for 8-bit dynamic range (L=255).
_C1 = (0.01 * 255) ** 2
_C2 = (0.03 * 255) ** 2


def _load_gray(path: str, size: int) -> np.ndarray:
    with Image.open(path) as img:
        gray = img.convert("L").resize((size, size), Image.BICUBIC)
        return np.asarray(gray, dtype=np.float64)


def _load_rgb(path: str, size: int) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((size, size), Image.BICUBIC)
        return np.asarray(rgb, dtype=np.uint8)


def ssim(reference_path: str, candidate_path: str, size: int = 256) -> float:
    """
    Global structural similarity between two images, on grayscale.

    Returns a value in [-1, 1] where 1 is identical. This is the global form of
    SSIM (one window over the whole image), which is cheaper than the windowed
    form and sufficient for ranking candidates against a fixed reference.
    """
    a = _load_gray(reference_path, size)
    b = _load_gray(candidate_path, size)

    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()

    numerator = (2 * mu_a * mu_b + _C1) * (2 * cov + _C2)
    denominator = (mu_a**2 + mu_b**2 + _C1) * (var_a + var_b + _C2)
    return float(numerator / denominator)


def histogram_correlation(reference_path: str, candidate_path: str, bins: int = 64, size: int = 256) -> float:
    """
    Pearson correlation between the two images' concatenated RGB histograms.

    Returns a value in [-1, 1]. Captures colour-palette agreement, which SSIM on
    grayscale misses entirely.
    """
    a = _load_rgb(reference_path, size)
    b = _load_rgb(candidate_path, size)

    def hist(arr: np.ndarray) -> np.ndarray:
        parts = [np.histogram(arr[:, :, c], bins=bins, range=(0, 255))[0] for c in range(3)]
        combined = np.concatenate(parts).astype(np.float64)
        total = combined.sum()
        return combined / total if total else combined

    ha, hb = hist(a), hist(b)
    if ha.std() == 0 or hb.std() == 0:
        return 0.0
    return float(np.corrcoef(ha, hb)[0, 1])


def compare(reference_path: str, candidate_path: str) -> Dict[str, Any]:
    """
    Compare two images and return the measured metrics.

    ``score`` is a 0-100 convenience blend of SSIM and histogram correlation,
    each first mapped from [-1, 1] to [0, 1]. It is a ranking aid, not a
    perceptual guarantee, and it is derived only from the two images given.
    """
    try:
        structural = ssim(reference_path, candidate_path)
        colour = histogram_correlation(reference_path, candidate_path)
    except (OSError, ValueError) as e:
        logger.warning("Could not compare %s to %s: %s", reference_path, candidate_path, e)
        return {"success": False, "error": str(e)}

    score = round(((structural + 1) / 2 * 0.6 + (colour + 1) / 2 * 0.4) * 100, 2)
    return {
        "success": True,
        "ssim": round(structural, 4),
        "histogram_correlation": round(colour, 4),
        "score": score,
        "metric": "0.6*SSIM + 0.4*histogram correlation, each rescaled to [0,1]",
    }
