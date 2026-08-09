"""
Master Execution Spine (agents/spine.py)

Provides centralized thread-safe sequential execution barriers, reentrant locking guards,
and pipeline state management to ensure non-overlapping execution across dependent agent tasks.
"""

import functools
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MasterSpineCoordinator:
    """
    Master Execution Spine for SenaAIgent.
    Manages global execution locks, pipeline barriers, and state conflict prevention.
    """

    _instance: Optional["MasterSpineCoordinator"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "MasterSpineCoordinator":
        """Singleton pattern for global spine coordination."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize MasterSpineCoordinator if not already initialized."""
        if getattr(self, "_initialized", False):
            return

        self._spine_lock = threading.RLock()
        self._execution_history: List[Dict[str, Any]] = []
        self._active_locks: Dict[str, str] = {}
        self._strict_sequential: bool = True
        self._lock_timeout: float = 30.0
        self._initialized = True
        logger.info("Master Execution Spine initialized successfully.")

    def set_strict_sequential(self, enabled: bool) -> None:
        """Enable or disable strict sequential spine execution."""
        with self._spine_lock:
            self._strict_sequential = enabled
            logger.info(f"Master Spine strict sequential mode set to: {enabled}")

    def acquire_lock(self, component_name: str, task_id: str) -> bool:
        """
        Acquire component lock on the spine.

        Args:
            component_name: Name of the pipeline subsystem (e.g. 'video', 'coder', 'healer').
            task_id: Unique task identifier.

        Returns:
            True if lock acquired, False otherwise.
        """
        with self._spine_lock:
            if component_name in self._active_locks:
                logger.warning(
                    f"Spine lock contention for {component_name}. Currently held by {self._active_locks[component_name]}"
                )
                if self._strict_sequential:
                    return False
            self._active_locks[component_name] = task_id
            self._execution_history.append({
                "timestamp": time.time(),
                "event": "lock_acquired",
                "component": component_name,
                "task_id": task_id,
            })
            return True

    def release_lock(self, component_name: str, task_id: str) -> bool:
        """
        Release component lock on the spine.

        Args:
            component_name: Name of the pipeline subsystem.
            task_id: Unique task identifier.

        Returns:
            True if lock released, False otherwise.
        """
        with self._spine_lock:
            if self._active_locks.get(component_name) == task_id:
                del self._active_locks[component_name]
                self._execution_history.append({
                    "timestamp": time.time(),
                    "event": "lock_released",
                    "component": component_name,
                    "task_id": task_id,
                })
                return True
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of Master Execution Spine.

        Returns:
            Dictionary with lock status and history.
        """
        with self._spine_lock:
            return {
                "spine_active": True,
                "strict_sequential": self._strict_sequential,
                "active_locks": dict(self._active_locks),
                "lock_count": len(self._active_locks),
                "history_length": len(self._execution_history),
                "recent_events": self._execution_history[-10:],
            }

    def spine_guarded(self, component_name: str):
        """
        Decorator to wrap any function execution inside a Master Spine lock.

        Args:
            component_name: Subsystem name.
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                task_id = f"{component_name}_{time.time()}"
                acquired = self.acquire_lock(component_name, task_id)
                if not acquired and self._strict_sequential:
                    raise RuntimeError(f"Spine Execution Locked: {component_name} is currently running another task.")
                try:
                    return func(*args, **kwargs)
                finally:
                    self.release_lock(component_name, task_id)
            return wrapper
        return decorator


# Global accessor function
def get_spine() -> MasterSpineCoordinator:
    """Get singleton instance of MasterSpineCoordinator."""
    return MasterSpineCoordinator()
