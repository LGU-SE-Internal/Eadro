from .base import DatasetMetadata
import polars as pl
from datetime import datetime
import numpy as np
from .common import smooth_sparse_data


def preaggregate_metrics(
    df: pl.DataFrame,
    start_time: datetime,
    end_time: datetime,
    metadata: DatasetMetadata,
) -> np.ndarray:
    """Pre-aggregate metrics data at second-level granularity for the entire time range"""
    total_seconds = int((end_time - start_time).total_seconds())
    num_services = len(metadata.services)
    num_metrics = len(metadata.metric_names)

    if df.height == 0:
        return np.zeros((num_services, total_seconds, num_metrics))

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
        return np.zeros((num_services, total_seconds, num_metrics))

    # Initialize with -1 to mark missing data (0 is a valid metric value)
    result = np.full((num_services, total_seconds, num_metrics), -1.0)

    # Aggregate all data at once
    metrics_stats = df_with_buckets.group_by(
        ["service_name", "metric", "time_bucket"]
    ).agg(pl.col("value").mean().alias("mean_value"))

    service_lookup = metadata.service_name_to_id
    metric_lookup = metadata.metric_name_to_id

    # Fill the result array
    for row in metrics_stats.iter_rows(named=True):
        service_name = row["service_name"]
        metric_name = row["metric"]
        time_bucket = row["time_bucket"]
        mean_value = row["mean_value"]

        service_id = service_lookup.get(service_name)
        metric_id = metric_lookup.get(metric_name)

        if (
            service_id is not None
            and metric_id is not None
            and 0 <= time_bucket < total_seconds
        ):
            metric_meta = metadata.metrics[metric_id]

            if metric_meta.max_value is not None and metric_meta.min_value is not None:
                # Check for valid range
                if np.isnan(metric_meta.max_value) or np.isnan(metric_meta.min_value):
                    # Invalid metadata, use raw value
                    normalized_value = mean_value
                elif metric_meta.max_value == metric_meta.min_value:
                    # All values are the same, normalize to 0.5 for stability
                    normalized_value = 0.5
                else:
                    value_range = metric_meta.max_value - metric_meta.min_value
                    normalized_value = (
                        mean_value - metric_meta.min_value
                    ) / value_range

                # Ensure the normalized value is finite
                if not np.isfinite(normalized_value):
                    normalized_value = 0.0

                result[service_id, time_bucket, metric_id] = normalized_value
            else:
                # No metadata available, use raw value
                result[service_id, time_bucket, metric_id] = mean_value

    # Apply smoothing to the entire time series
    smooth_sparse_data(result, num_services, total_seconds, num_metrics)

    # Replace any remaining -1 markers with 0 (missing data)
    result[result == -1.0] = 0.0

    return result
