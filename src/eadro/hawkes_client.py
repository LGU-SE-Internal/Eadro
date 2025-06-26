"""
Hawkes Process RPC Client

This module provides a client for calling the Hawkes process modeling service.
"""

import logging
import time
from typing import List, Optional, Dict, Any
import numpy as np
import requests
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class HawkesRPCClient:
    """Client for communicating with the Hawkes process modeling service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the HTTP session."""
        if hasattr(self, "_session"):
            self._session.close()

    def _serialize_data(self, data: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        else:
            return data

    def _deserialize_data(self, data: Any) -> Any:
        """Convert lists back to numpy arrays."""
        if isinstance(data, list) and len(data) > 0:
            # Check if this looks like a numeric array
            try:
                return np.array(data, dtype=np.float64)
            except (ValueError, TypeError):
                return data
        return data

    def health_check(self) -> bool:
        """Check if the Hawkes service is healthy."""
        try:
            response = self._session.get(
                f"{self.base_url}/health", timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
        return False

    def wait_for_service(self, max_wait_time: int = 60) -> bool:
        """Wait for the service to become available."""
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            if self.health_check():
                logger.info("Hawkes service is ready")
                return True
            logger.info("Waiting for Hawkes service to become ready...")
            time.sleep(2)

        logger.error(
            f"Hawkes service did not become ready within {max_wait_time} seconds"
        )
        return False

    def model_hawkes_process(
        self,
        chunk_logs: List[np.ndarray],
        end_time: float,
        event_num: int,
        decay: float = 3,
        ini_intensity: float = 0.2,
    ) -> Optional[np.ndarray]:
        """
        Apply Hawkes process modeling to log events.

        Args:
            chunk_logs: List of arrays containing event timestamps for each event type
            end_time: End time of the chunk
            event_num: Number of event types
            decay: Decay parameter for Hawkes process
            ini_intensity: Initial intensity for baseline

        Returns:
            Array of baseline intensities for each event type, or None if failed
        """
        payload = {
            "chunk_logs": self._serialize_data(chunk_logs),
            "end_time": float(end_time),
            "event_num": int(event_num),
            "decay": float(decay),
            "ini_intensity": float(ini_intensity),
        }

        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    f"{self.base_url}/hawkes/model", json=payload, timeout=self.timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        baseline = self._deserialize_data(data["baseline"])
                        method = data.get("method", "unknown")
                        logger.debug(
                            f"Hawkes modeling successful using method: {method}"
                        )
                        return baseline
                    else:
                        logger.error(f"Hawkes modeling failed: {data.get('error')}")
                        return None
                else:
                    logger.error(f"HTTP error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Request timeout (attempt {attempt + 1}/{self.max_retries})"
                )
            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"Connection error (attempt {attempt + 1}/{self.max_retries})"
                )
            except Exception as e:
                logger.error(f"Unexpected error in hawkes modeling: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        logger.error("All retry attempts failed for Hawkes modeling")
        return None

    def model_hawkes_batch(
        self, batches: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Apply Hawkes process modeling to multiple chunks in batch.

        Args:
            batches: List of batch configurations, each containing:
                - chunk_logs: List of event timestamp arrays
                - end_time: End time of the chunk
                - event_num: Number of event types
                - decay: Decay parameter (optional)
                - ini_intensity: Initial intensity (optional)

        Returns:
            List of results for each batch, or None if failed
        """
        # Serialize the batch data
        serialized_batches = []
        for batch in batches:
            serialized_batch = {
                "chunk_logs": self._serialize_data(batch["chunk_logs"]),
                "end_time": float(batch["end_time"]),
                "event_num": int(batch["event_num"]),
                "decay": float(batch.get("decay", 3)),
                "ini_intensity": float(batch.get("ini_intensity", 0.2)),
            }
            serialized_batches.append(serialized_batch)

        payload = {"batches": serialized_batches}

        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    f"{self.base_url}/hawkes/batch",
                    json=payload,
                    timeout=self.timeout * 2,  # Give more time for batch processing
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        results = data["results"]
                        # Deserialize baseline arrays in results
                        for result in results:
                            if result.get("success") and "baseline" in result:
                                result["baseline"] = self._deserialize_data(
                                    result["baseline"]
                                )
                        return results
                    else:
                        logger.error(
                            f"Batch hawkes modeling failed: {data.get('error')}"
                        )
                        return None
                else:
                    logger.error(f"HTTP error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Batch request timeout (attempt {attempt + 1}/{self.max_retries})"
                )
            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"Connection error (attempt {attempt + 1}/{self.max_retries})"
                )
            except Exception as e:
                logger.error(f"Unexpected error in batch hawkes modeling: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        logger.error("All retry attempts failed for batch Hawkes modeling")
        return None


@contextmanager
def hawkes_client(host: str = "localhost", port: int = 8080, **kwargs):
    """Context manager for Hawkes RPC client."""
    client = HawkesRPCClient(host=host, port=port, **kwargs)
    try:
        yield client
    finally:
        client.close()
