import numpy as np


def smooth_sparse_data(
    data: np.ndarray, num_services: int, interval: int, num_features: int
):
    """
    Smooth sparse data using interpolation.
    Note: This function now treats -1 as missing data marker instead of 0,
    since 0 can be a valid metric value (e.g., 0% CPU usage).
    """
    for service_id in range(num_services):
        for feature_id in range(num_features):
            # Get the time series for this service-feature combination
            time_series = data[service_id, :, feature_id]

            # Find non-missing indices (using -1 as missing data marker)
            non_missing_indices = np.where(time_series != -1)[0]

            if len(non_missing_indices) == 0:
                # No data points, set all to 0 (default value)
                data[service_id, :, feature_id] = 0.0
                continue
            elif len(non_missing_indices) == 1:
                # Only one data point, forward fill
                single_value = time_series[non_missing_indices[0]]
                data[service_id, non_missing_indices[0] :, feature_id] = single_value
            else:
                # Multiple data points, use linear interpolation
                for i in range(len(non_missing_indices) - 1):
                    start_idx = non_missing_indices[i]
                    end_idx = non_missing_indices[i + 1]
                    start_val = time_series[start_idx]
                    end_val = time_series[end_idx]

                    # Linear interpolation between data points
                    if end_idx - start_idx > 1:
                        interpolated_values = np.linspace(
                            start_val, end_val, end_idx - start_idx + 1
                        )
                        data[service_id, start_idx : end_idx + 1, feature_id] = (
                            interpolated_values
                        )

                # Forward fill from the last data point to the end
                last_idx = non_missing_indices[-1]
                last_val = time_series[last_idx]
                data[service_id, last_idx:, feature_id] = last_val

                # Backward fill from the first data point to the beginning
                first_idx = non_missing_indices[0]
                first_val = time_series[first_idx]
                data[service_id, : first_idx + 1, feature_id] = first_val
