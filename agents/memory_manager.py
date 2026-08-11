"""
Reclaim GPU and process memory held by loaded pipelines.

This process can hold three large models at once: SD 1.5 for stills, Stable Video
Diffusion at roughly 8GB, and AnimateDiff's motion module over another SD 1.5
UNet. They are cached deliberately, because reloading costs seconds on every
request, but nothing ever released them, so a session that touched all three kept
all three resident until the server was restarted.

Releasing is explicit rather than automatic on a timer: dropping a pipeline that
is about to be used again is worse than holding it. Callers release before
loading a different heavy pipeline, or on request.
"""

import gc
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False

try:
    import psutil
    _process = psutil.Process()
except Exception:  # pragma: no cover - environment dependent
    _process = None


def _rss_mb() -> Optional[float]:
    if _process is None:
        return None
    try:
        return round(_process.memory_info().rss / (1024 * 1024), 1)
    except Exception:  # pragma: no cover
        return None


def empty_gpu_cache() -> bool:
    """Return allocator-held GPU blocks to the system. True if anything ran."""
    if not TORCH_AVAILABLE:
        return False
    freed = False
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
            freed = True
        except Exception as e:  # pragma: no cover - backend dependent
            logger.warning("torch.mps.empty_cache() failed: %s", e)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            freed = True
        except Exception as e:  # pragma: no cover
            logger.warning("torch.cuda.empty_cache() failed: %s", e)
    return freed


def release(*engines: Any, keep: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Drop cached pipelines from the given engines and reclaim memory.

    Args:
        engines: Objects with a ``pipe`` attribute, such as PyTorchDiffusionEngine
            or PyTorchVideoDiffusionEngine. Anything without one is ignored, so a
            caller can pass a mixed bag without checking types.
        keep: Class names to leave loaded, for releasing everything except the one
            about to be used.

    Returns:
        What was released and the resident set size before and after, measured
        rather than asserted.
    """
    keep = set(keep or [])
    before = _rss_mb()
    released: List[str] = []

    for engine in engines:
        name = type(engine).__name__
        if name in keep or not hasattr(engine, "pipe"):
            continue
        if getattr(engine, "pipe", None) is None:
            continue
        try:
            engine.pipe = None
            if hasattr(engine, "initialized"):
                engine.initialized = False
            if hasattr(engine, "active_lora"):
                engine.active_lora = None
            released.append(name)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not release %s: %s", name, e)

    collected = gc.collect()
    cache_emptied = empty_gpu_cache()
    after = _rss_mb()

    result = {
        "released": released,
        "gc_objects_collected": collected,
        "gpu_cache_emptied": cache_emptied,
        "rss_mb_before": before,
        "rss_mb_after": after,
        "rss_mb_freed": round(before - after, 1) if (before is not None and after is not None) else None,
    }
    if released:
        logger.info("Released %s; RSS %s -> %s MB", ", ".join(released), before, after)
    return result


def snapshot() -> Dict[str, Any]:
    """Current process memory, for deciding whether a release is worth doing."""
    total_mb = None
    if _process is not None:
        try:
            import psutil as _p
            total_mb = round(_p.virtual_memory().total / (1024 * 1024), 1)
        except Exception:  # pragma: no cover
            pass
    return {"rss_mb": _rss_mb(), "system_total_mb": total_mb, "pid": os.getpid()}
