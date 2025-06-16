import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Iterator, Tuple, List


class ExpiringDict:
    """
    A thread-safe dictionary with automatic expiration of entries.
    Uses OrderedDict to efficiently manage expiration based on insertion time.
    """

    def __init__(self, expire_seconds: float = 600, cleanup_interval: int = 10):
        """
        Initialize the expiring dictionary.

        Args:
            expire_seconds: Time in seconds after which entries expire
            cleanup_interval: Number of operations between cleanup checks
        """
        self._data: OrderedDict[Any, Any] = OrderedDict()  # key -> (timestamp, value)
        self._expire_seconds = expire_seconds
        self._cleanup_interval = cleanup_interval
        self._operation_count = 0
        self._lock = threading.RLock()  # Use RLock for nested locking

    def _cleanup_expired(self, current_time: Optional[float] = None) -> None:
        """Remove expired entries from the dictionary."""
        if current_time is None:
            current_time = time.time()

        cutoff_time = current_time - self._expire_seconds

        # Remove expired items from the beginning (oldest first)
        expired_keys = []
        for key, (timestamp, _) in self._data.items():
            if timestamp < cutoff_time:
                expired_keys.append(key)
            else:
                # Since OrderedDict maintains insertion order,
                # we can break once we find a non-expired item
                break

        for key in expired_keys:
            del self._data[key]

    def _maybe_cleanup(self) -> None:
        """Conditionally trigger cleanup based on operation count."""
        self._operation_count += 1
        if self._operation_count >= self._cleanup_interval:
            self._cleanup_expired()
            self._operation_count = 0

    def set(self, key: Any, value: Any) -> None:
        """Set a key-value pair with current timestamp."""
        current_time = time.time()
        with self._lock:
            self._data[key] = (current_time, value)
            self._maybe_cleanup()

    def get(self, key: Any, default: Any = None) -> Any:
        """Get value by key, returning default if not found or expired."""
        with self._lock:
            if key in self._data:
                timestamp, value = self._data[key]
                return value
            else:
                return default

    def __setitem__(self, key: Any, value: Any) -> None:
        self.set(key, value)

    def __getitem__(self, key: Any) -> Any:
        value = self.get(key)
        if value is None and key not in self._data:
            raise KeyError(key)
        return value

    def __delitem__(self, key: Any) -> None:
        with self._lock:
            if key not in self._data:
                raise KeyError(key)
            del self._data[key]

    def __contains__(self, key: Any) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            self._cleanup_expired()
            return len(self._data)

    def keys(self) -> List[Any]:
        """Return list of all non-expired keys."""
        with self._lock:
            self._cleanup_expired()
            return list(self._data.keys())

    def values(self) -> List[Any]:
        """Return list of all non-expired values."""
        with self._lock:
            self._cleanup_expired()
            return [value for _, value in self._data.values()]

    def items(self) -> List[Tuple[Any, Any]]:
        """Return list of all non-expired (key, value) pairs."""
        with self._lock:
            self._cleanup_expired()
            return [(key, value) for key, (_, value) in self._data.items()]

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._data.clear()
            self._operation_count = 0

    def cleanup(self) -> int:
        """Manually trigger cleanup and return number of expired items removed."""
        with self._lock:
            original_size = len(self._data)
            self._cleanup_expired()
            return original_size - len(self._data)
