"""
Time estimates for renders, built from what this machine has actually done.

A first-order estimate is only useful if it comes from measurement. Rather than
publishing a constant that drifts out of date the moment anything changes, this
records the seconds-per-frame of every completed render and estimates from that
history. Before a given configuration has ever run, it says so instead of
guessing.

Estimates are kept per (engine, pixel count) because cost scales with area and
differs by an order of magnitude between the diffusion pipeline and the
procedural fallback. A run is folded in with an exponential moving average so a
recent measurement counts for more than an old one without a single outlier
throwing the figure.
"""

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CALIBRATION_PATH = os.environ.get("RENDER_CALIBRATION_PATH", "config/render_calibration.json")

# Weight of the newest sample. 0.3 tracks a real change within a few runs while
# still smoothing noise from thermal throttling or a busy machine.
_EWMA_ALPHA = 0.3

_lock = threading.Lock()


def _bucket(engine: str, width: int, height: int) -> str:
    return f"{engine}@{width}x{height}"


def _load() -> Dict[str, Any]:
    try:
        with open(CALIBRATION_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(CALIBRATION_PATH))
    if directory:
        os.makedirs(directory, exist_ok=True)
    try:
        with open(CALIBRATION_PATH, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
    except OSError as e:  # pragma: no cover - disk dependent
        logger.warning("Could not write render calibration: %s", e)


def record(engine: str, width: int, height: int, frames: int, elapsed_seconds: float) -> None:
    """Fold one completed render into the calibration for its configuration."""
    if frames <= 0 or elapsed_seconds <= 0:
        return
    per_frame = elapsed_seconds / frames
    key = _bucket(engine, width, height)

    with _lock:
        data = _load()
        entry = data.get(key)
        if entry:
            entry["seconds_per_frame"] = round(
                (1 - _EWMA_ALPHA) * entry["seconds_per_frame"] + _EWMA_ALPHA * per_frame, 4)
            entry["samples"] = entry.get("samples", 1) + 1
        else:
            entry = {"seconds_per_frame": round(per_frame, 4), "samples": 1}
        entry["last_frames"] = frames
        entry["last_elapsed_seconds"] = round(elapsed_seconds, 2)
        data[key] = entry
        _save(data)

    logger.info("Render calibration %s: %.3fs/frame over %d sample(s)",
                key, entry["seconds_per_frame"], entry["samples"])


def estimate(engine: str, width: int, height: int, frames: int) -> Dict[str, Any]:
    """
    Estimated seconds for a render, or an explicit unknown.

    Falls back to scaling a measurement taken at a different resolution by pixel
    area, which is a rough but honest first order, and says which it did.
    """
    data = _load()
    key = _bucket(engine, width, height)
    entry = data.get(key)

    if entry:
        seconds = entry["seconds_per_frame"] * frames
        return {
            "estimate_seconds": round(seconds, 1),
            "seconds_per_frame": entry["seconds_per_frame"],
            "basis": "measured at this exact resolution",
            "samples": entry["samples"],
            "confidence": "high" if entry["samples"] >= 3 else "low, one or two samples so far",
        }

    # Nothing at this size: scale the nearest same-engine measurement by area.
    same_engine = {k: v for k, v in data.items() if k.startswith(f"{engine}@")}
    if same_engine:
        other_key, other = next(iter(sorted(same_engine.items(), key=lambda kv: -kv[1]["samples"])))
        try:
            dims = other_key.split("@", 1)[1]
            ow, oh = (int(part) for part in dims.split("x"))
            ratio = (width * height) / max(1, ow * oh)
        except (ValueError, IndexError):
            ratio = 1.0
        seconds = other["seconds_per_frame"] * ratio * frames
        return {
            "estimate_seconds": round(seconds, 1),
            "seconds_per_frame": round(other["seconds_per_frame"] * ratio, 4),
            "basis": f"scaled by pixel area from {other_key}",
            "samples": other["samples"],
            "confidence": "rough, never measured at this resolution",
        }

    return {
        "estimate_seconds": None,
        "basis": f"no render has been measured for {engine} yet",
        "samples": 0,
        "confidence": "none",
    }


def calibration() -> Dict[str, Any]:
    """Everything measured so far, for display."""
    return {"path": CALIBRATION_PATH, "entries": _load()}
