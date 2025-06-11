import threading
import time


class DoubleBufferExpiringDict:
    def __init__(self, expire_seconds: float = 600):
        self._buffer1 = {}
        self._buffer2 = {}
        self._active_buffer = self._buffer1
        self._inactive_buffer = self._buffer2
        self._buffer_start_time = time.time()
        self._expire = expire_seconds
        self._lock = threading.Lock()
        self._set_count = 0

    def _switch_buffer_if_needed(self):
        now = time.time()
        if now - self._buffer_start_time > self._expire:
            self._inactive_buffer.clear()
            self._inactive_buffer, self._active_buffer = (
                self._active_buffer,
                self._inactive_buffer
            )
            self._buffer_start_time = now

    def set(self, key, value):
        with self._lock:
            self._set_count += 1
            if self._set_count >= 10:
                self._switch_buffer_if_needed()
                self._set_count = 0
            self._active_buffer[key] = value

    def delete(self, key):
        with self._lock:
            self._active_buffer.pop(key, None)
            self._inactive_buffer.pop(key, None)

    def get(self, key, default=None):
        with self._lock:
            if key in self._active_buffer:
                return self._active_buffer[key]
            if key in self._inactive_buffer:
                return self._inactive_buffer[key]
            print(f"WARNING! key {key} not found in active or inactive buffer")
            return default

    def __setitem__(self, key, value):
        self.set(key, value)

    def __getitem__(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def keys(self):
        with self._lock:
            return list(set(self._active_buffer.keys()) | set(self._inactive_buffer.keys()))

    def values(self):
        with self._lock:
            keys = set(self._active_buffer.keys()) | set(self._inactive_buffer.keys())
            return [self.get(k) for k in keys]

    def items(self):
        with self._lock:
            keys = set(self._active_buffer.keys()) | set(self._inactive_buffer.keys())
            return [(k, self.get(k)) for k in keys]

    def __contains__(self, key):
        with self._lock:
            return key in self._active_buffer or key in self._inactive_buffer

    def __len__(self):
        with self._lock:
            return len(set(self._active_buffer.keys()) | set(self._inactive_buffer.keys()))