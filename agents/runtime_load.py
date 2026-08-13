"""
Actual load on this process.

The dashboard gauge was fed by the orchestrator's task queue, but nothing in the
app enqueues to it: image renders call the diffusion engine directly and chat
calls the agent directly, so the queue is empty and every agent reads "available"
no matter how hard the machine is working. The arithmetic was honest and the
subject was wrong — the gauge sat at 0% through a 50-step render.

This measures work the process is really doing: requests in flight right now, and
CPU and memory sampled from the process itself.
"""

import threading
import time
from typing import Any, Dict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    PSUTIL_AVAILABLE = False

# Weighting for the headline score. In-flight work dominates because one render
# saturating the GPU matters more to a user than the CPU percentage it happens to
# show, and MPS utilisation is not readable from psutil.
_INFLIGHT_WEIGHT = 0.7
_CPU_WEIGHT = 0.3

# What counts as "fully loaded" for scoring. Two concurrent renders already make
# this machine unresponsive, so the scale is deliberately short.
_SATURATION_INFLIGHT = 2

_lock = threading.Lock()
_in_flight: Dict[str, int] = {}
_started_at: Dict[str, float] = {}
_process = psutil.Process() if PSUTIL_AVAILABLE else None


class track:
    """
    Context manager counting one unit of work of the given kind as in flight.

    Reentrant across threads; the counter is guarded and never drops below zero
    even if a caller unbalances it.

        with track("diffusion"):
            ...
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._start = None

    def __enter__(self):
        self._start = time.time()
        with _lock:
            _in_flight[self.kind] = _in_flight.get(self.kind, 0) + 1
            _started_at.setdefault(self.kind, self._start)
        return self

    def __exit__(self, exc_type, exc, tb):
        with _lock:
            remaining = max(0, _in_flight.get(self.kind, 0) - 1)
            _in_flight[self.kind] = remaining
            if remaining == 0:
                _started_at.pop(self.kind, None)
        return False


def reset() -> None:
    """Clear all counters. For tests."""
    with _lock:
        _in_flight.clear()
        _started_at.clear()


def in_flight() -> Dict[str, int]:
    """Copy of the per-kind in-flight counts, omitting kinds sitting at zero."""
    with _lock:
        return {kind: n for kind, n in _in_flight.items() if n > 0}


def snapshot() -> Dict[str, Any]:
    """
    Current load.

    ``cpu_percent`` is measured since the previous call, which is how psutil
    reports it without blocking. The first call after startup therefore reads
    0.0; the dashboard polls every few seconds, so subsequent reads are real.
    """
    active = in_flight()
    total_in_flight = sum(active.values())

    cpu = memory_mb = memory_percent = None
    if _process is not None:
        try:
            cpu = round(_process.cpu_percent(None), 1)
            memory_mb = round(_process.memory_info().rss / (1024 * 1024), 1)
            memory_percent = round(_process.memory_percent(), 1)
        except Exception:  # pragma: no cover - psutil can fail transiently
            pass

    inflight_ratio = min(1.0, total_in_flight / _SATURATION_INFLIGHT)
    # psutil reports process CPU as a percentage of one core, so it exceeds 100
    # on a multi-core machine; clamp before folding it into a 0-100 score.
    cpu_ratio = min(1.0, (cpu or 0.0) / 100.0)
    score = round((inflight_ratio * _INFLIGHT_WEIGHT + cpu_ratio * _CPU_WEIGHT) * 100, 1)

    if score >= 80:
        level, color = "critical", "#ef4444"
    elif score >= 60:
        level, color = "high", "#f59e0b"
    elif score >= 30:
        level, color = "moderate", "#eab308"
    else:
        level, color = "low", "#22c55e"

    longest = 0.0
    now = time.time()
    with _lock:
        if _started_at:
            longest = round(now - min(_started_at.values()), 1)

    return {
        "score": score,
        "level": level,
        "color": color,
        "in_flight": total_in_flight,
        "in_flight_by_kind": active,
        "longest_running_seconds": longest,
        "cpu_percent": cpu,
        "memory_mb": memory_mb,
        "memory_percent": memory_percent,
        "psutil_available": PSUTIL_AVAILABLE,
    }
