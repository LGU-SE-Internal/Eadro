from typing import Any, Callable, Generic, Optional, TypeVar
from pathlib import Path
import threading
import pickle
import logging

T = TypeVar("T")


class CacheManager(Generic[T]):
    """
    A generic, thread-safe manager for file-based object caching using pickle.
    """

    def __init__(self, cache_path: Path):
        self.cache_file = cache_path
        self.cache_dir = self.cache_file.parent
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, T] = self._load_cache()
        self._lock = threading.Lock()

    def _load_cache(self) -> dict[str, T]:
        """Loads the cache from a pickle file if it exists."""
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, "rb") as f:
                cache = pickle.load(f)
            logging.info(f"Loaded {len(cache)} items from cache: {self.cache_file}")
            return cache
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            logging.warning(
                f"Failed to load cache {self.cache_file}: {e}. Starting fresh."
            )
            return {}

    def save(self):
        """Saves the current cache to a pickle file."""
        with self._lock:
            try:
                with open(self.cache_file, "wb") as f:
                    pickle.dump(self._cache, f)
                logging.info(
                    f"Saved {len(self._cache)} items to cache: {self.cache_file}"
                )
            except Exception as e:
                logging.error(f"Failed to save cache {self.cache_file}: {e}")

    def get(self, key: str) -> Optional[T]:
        """Gets an item from the cache by key."""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: T):
        """Sets an item in the cache."""
        with self._lock:
            self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def get_or_compute(self, key: str, compute_fn: Callable[[], T]) -> T:
        """
        Retrieves an item from the cache. If not found, it computes the value,
        stores it in the cache, and then returns it.
        """
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value

        new_value = compute_fn()
        self.set(key, new_value)
        return new_value
