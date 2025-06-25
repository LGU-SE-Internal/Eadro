import os
import logging
import pickle
import json
import hashlib
import random
import numpy as np
import torch
import resource
from datetime import datetime, timedelta
from typing import Any, Callable, Generic, Optional, TypeVar
from pathlib import Path
import threading
from functools import wraps
from pprint import pformat
import inspect
import sys
from enum import Enum, auto


class Dataset(Enum):
    EADRO_SOCIAL_NETWORK = auto()
    EADRO_TRAIN_TICKET = auto()
    RCABENCH = auto()


T = TypeVar("T")


def load_chunks(data_dir):
    logging.info("Load from {}".format(data_dir))
    with open(os.path.join(data_dir, "chunk_train.pkl"), "rb") as fr:
        chunk_train = pickle.load(fr)
    with open(os.path.join(data_dir, "chunk_test.pkl"), "rb") as fr:
        chunk_test = pickle.load(fr)
    return chunk_train, chunk_test


def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith(".json")
        with open(filepath, "r") as f:
            return json.loads(f.read())
    else:
        logging.error("File path " + filepath + " not exists!")
        return


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def dump_scores(result_dir, hash_id, scores, converge):
    with open(os.path.join(result_dir, "experiments.txt"), "a+") as fw:
        fw.write(
            hash_id
            + ": "
            + (datetime.now() + timedelta(hours=8)).strftime("%Y/%m/%d-%H:%M:%S")
            + "\n"
        )
        fw.write(
            "* Test result -- "
            + "\t".join(["{}:{:.4f}".format(k, v) for k, v in scores.items()])
            + "\n"
        )
        fw.write("Best score got at epoch: " + str(converge) + "\n")
        fw.write("{}{}".format("=" * 40, "\n"))


def dump_params(params):
    hash_id = hashlib.md5(
        str(sorted([(k, v) for k, v in params.items()])).encode("utf-8")
    ).hexdigest()[0:8]
    result_dir = os.path.join(params["result_dir"], hash_id)
    os.makedirs(result_dir, exist_ok=True)

    json_pretty_dump(params, os.path.join(result_dir, "params.json"))

    log_file = os.path.join(result_dir, "running.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return hash_id


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def timeit(*, log_level: str = "DEBUG", log_args: bool | set[str] = True):
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__qualname__:<20}"

            sys.stdout.flush()

            # Get initial memory usage
            initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            start = datetime.now()
            result = func(*args, **kwargs)
            end = datetime.now()

            # Get peak memory usage
            maxrss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            maxrss_mib = maxrss_kib / 1024

            duration = end - start
            duration_message = f"duration={duration.total_seconds():.6f}s"
            memory_message = f"peak_memory={maxrss_mib:.3f}MiB"
            print(f"exit  {func_name} {duration_message} {memory_message}")
            sys.stdout.flush()

            return result

        return wrapper

    return decorator
