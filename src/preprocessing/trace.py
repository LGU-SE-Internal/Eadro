from .base import DatasetMetadata

import polars as pl
from datetime import datetime
import numpy as np
from .common import smooth_sparse_data


def preaggregate_traces(
    df: pl.DataFrame,
    start_time: datetime,
    end_time: datetime,
    metadata: DatasetMetadata,
) -> np.ndarray:
    """Pre-aggregate trace data at second-level granularity for the entire time range"""
    total_seconds = int((end_time - start_time).total_seconds())
    num_services = len(metadata.services)
    expected_features = 2  # latency and invocation count

    if df.height == 0:
        return np.zeros((num_services, total_seconds, expected_features))

    # Add second-level time buckets
    df_with_buckets = df.with_columns(
        [
            ((pl.col("time") - start_time).dt.total_seconds())
            .floor()
            .cast(pl.Int32)
            .alias("time_bucket")
        ]
    ).filter((pl.col("time_bucket") >= 0) & (pl.col("time_bucket") < total_seconds))

    if df_with_buckets.height == 0:
        return np.zeros((num_services, total_seconds, expected_features))

    # Initialize with -1 to mark missing data (0 is valid for traces too)
    result = np.full((num_services, total_seconds, expected_features), -1.0)

    # Calculate latency and invocation statistics
    trace_stats = df_with_buckets.group_by(["service_name", "time_bucket"]).agg(
        [
            pl.col("duration").mean().alias("avg_latency"),
            pl.col("duration").count().alias("invocation_count"),
        ]
    )

    service_lookup = metadata.service_name_to_id

    for row in trace_stats.iter_rows(named=True):
        service_name = row["service_name"]
        time_bucket = row["time_bucket"]
        avg_latency = row["avg_latency"]
        invocation_count = row["invocation_count"]

        service_id = service_lookup.get(service_name)
        if service_id is not None and 0 <= time_bucket < total_seconds:
            # Handle potential None values and ensure finite values
            latency_val = (
                avg_latency
                if avg_latency is not None and np.isfinite(avg_latency)
                else 0.0
            )
            count_val = (
                invocation_count
                if invocation_count is not None and np.isfinite(invocation_count)
                else 0.0
            )

            result[service_id, time_bucket, 0] = latency_val
            result[service_id, time_bucket, 1] = count_val

    # Apply smoothing
    smooth_sparse_data(result, num_services, total_seconds, expected_features)

    # Replace any remaining -1 markers with 0 (missing data)
    result[result == -1.0] = 0.0

    # Normalize latency using metadata statistics
    service_duration_ranges = {}
    for trace_meta in metadata.traces:
        service_name = trace_meta.service_name
        if service_name not in service_duration_ranges:
            service_duration_ranges[service_name] = {
                "min": trace_meta.min_duration,
                "max": trace_meta.max_duration,
            }
        else:
            if trace_meta.min_duration is not None:
                if service_duration_ranges[service_name]["min"] is None:
                    service_duration_ranges[service_name]["min"] = (
                        trace_meta.min_duration
                    )
                else:
                    service_duration_ranges[service_name]["min"] = min(
                        service_duration_ranges[service_name]["min"],
                        trace_meta.min_duration,
                    )

            if trace_meta.max_duration is not None:
                if service_duration_ranges[service_name]["max"] is None:
                    service_duration_ranges[service_name]["max"] = (
                        trace_meta.max_duration
                    )
                else:
                    service_duration_ranges[service_name]["max"] = max(
                        service_duration_ranges[service_name]["max"],
                        trace_meta.max_duration,
                    )

    # Apply normalization for each service
    for service_name, duration_range in service_duration_ranges.items():
        service_id = service_lookup.get(service_name)
        if service_id is not None:
            global_min = duration_range["min"]
            global_max = duration_range["max"]

            if (
                global_min is not None
                and global_max is not None
                and global_max > global_min
                and np.isfinite(global_min)
                and np.isfinite(global_max)
            ):
                service_latencies = result[service_id, :, 0]
                service_latencies = np.clip(service_latencies, global_min, global_max)
                normalized_latencies = (service_latencies - global_min) / (
                    global_max - global_min
                )

                # Ensure normalized values are finite
                normalized_latencies = np.nan_to_num(
                    normalized_latencies, nan=0.0, posinf=1.0, neginf=0.0
                )
                result[service_id, :, 0] = normalized_latencies

    return result
