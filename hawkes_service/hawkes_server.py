#!/usr/bin/env python3
"""
Hawkes Process Modeling RPC Server using Python 3.8 and tick library.

This service provides Hawkes process modeling functionality through a simple HTTP API.
"""

import traceback
import numpy as np
from flask import Flask, request, jsonify
from tick.hawkes import HawkesADM4
from loguru import logger

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=""),
    format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}",
    level="INFO",
)

app = Flask(__name__)


def serialize_array(arr):
    """Convert numpy array to list for JSON serialization."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def deserialize_array(data):
    """Convert list back to numpy array."""
    if isinstance(data, list):
        return np.array(data)
    return data


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "hawkes-modeling"})


@app.route("/hawkes/model", methods=["POST"])
def hawkes_modeling():
    """
    Apply Hawkes process modeling to log events.

    Expected JSON payload:
    {
        "chunk_logs": [[timestamps_for_event_0], [timestamps_for_event_1], ...],
        "end_time": float,
        "event_num": int,
        "decay": float (optional, default=3),
        "ini_intensity": float (optional, default=0.2)
    }

    Returns:
    {
        "success": bool,
        "baseline": [float, ...],
        "error": str (if success=false)
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        # Extract parameters
        chunk_logs = data.get("chunk_logs", [])
        end_time = data.get("end_time")
        event_num = data.get("event_num")
        decay = data.get("decay", 3)
        ini_intensity = data.get("ini_intensity", 0.2)

        # Validate required parameters
        if not chunk_logs or end_time is None or event_num is None:
            return jsonify(
                {
                    "success": False,
                    "error": "Missing required parameters: chunk_logs, end_time, event_num",
                }
            ), 400

        # Convert chunk_logs to numpy arrays
        chunk_logs_np = []
        for i, knots in enumerate(chunk_logs):
            if isinstance(knots, list):
                chunk_logs_np.append(np.array(knots, dtype=np.float64))
            else:
                chunk_logs_np.append(np.array(knots, dtype=np.float64))

        # Validate that we have the expected number of event types
        if len(chunk_logs_np) != event_num:
            logger.warning(
                f"Event number mismatch: expected {event_num}, got {len(chunk_logs_np)}"
            )
            # Pad or trim to match expected event_num
            while len(chunk_logs_np) < event_num:
                chunk_logs_np.append(np.array([0.0]))
            chunk_logs_np = chunk_logs_np[:event_num]

        # Check if all event types are empty
        total_events = sum(len(knots) for knots in chunk_logs_np)
        if total_events == 0:
            logger.info("No events found, returning zero baseline")
            baseline = np.zeros(event_num)
            return jsonify(
                {
                    "success": True,
                    "baseline": serialize_array(baseline),
                    "method": "fallback_empty",
                }
            )

        # Apply Hawkes modeling
        try:
            logger.info(
                f"Applying Hawkes modeling with decay={decay}, ini_intensity={ini_intensity}"
            )
            model = HawkesADM4(decay)

            # Fit the model
            model.fit(
                chunk_logs_np,
                end_time,
                baseline_start=np.ones(event_num) * ini_intensity,
            )

            baseline = np.array(model.baseline)

            logger.info(f"Hawkes modeling successful, baseline shape: {baseline.shape}")

            return jsonify(
                {
                    "success": True,
                    "baseline": serialize_array(baseline),
                    "method": "hawkes",
                }
            )

        except Exception as hawkes_error:
            logger.warning(
                f"Hawkes modeling failed: {hawkes_error}, falling back to simple counting"
            )

            # Fallback: simple counting
            baseline = np.zeros(event_num)
            for i, knots in enumerate(chunk_logs_np):
                if len(knots) > 0:
                    baseline[i] = len(knots) / end_time

            return jsonify(
                {
                    "success": True,
                    "baseline": serialize_array(baseline),
                    "method": "fallback_counting",
                    "hawkes_error": str(hawkes_error),
                }
            )

    except Exception as e:
        logger.error(f"Error in hawkes_modeling: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@app.route("/hawkes/batch", methods=["POST"])
def hawkes_batch_modeling():
    """
    Apply Hawkes process modeling to multiple chunks in batch.

    Expected JSON payload:
    {
        "batches": [
            {
                "chunk_logs": [[timestamps_for_event_0], [timestamps_for_event_1], ...],
                "end_time": float,
                "event_num": int,
                "decay": float (optional),
                "ini_intensity": float (optional)
            },
            ...
        ]
    }

    Returns:
    {
        "success": bool,
        "results": [{"baseline": [...], "method": "..."}, ...],
        "error": str (if success=false)
    }
    """
    try:
        data = request.get_json()

        if not data or "batches" not in data:
            return jsonify({"success": False, "error": "No batches provided"}), 400

        batches = data["batches"]
        results = []

        for i, batch in enumerate(batches):
            try:
                # Process each batch individually
                chunk_logs = batch.get("chunk_logs", [])
                end_time = batch.get("end_time")
                event_num = batch.get("event_num")
                decay = batch.get("decay", 3)
                ini_intensity = batch.get("ini_intensity", 0.2)

                if not chunk_logs or end_time is None or event_num is None:
                    results.append(
                        {
                            "success": False,
                            "error": f"Batch {i}: Missing required parameters",
                        }
                    )
                    continue

                # Convert and process
                chunk_logs_np = []
                for knots in chunk_logs:
                    chunk_logs_np.append(np.array(knots, dtype=np.float64))

                # Pad or trim to match expected event_num
                while len(chunk_logs_np) < event_num:
                    chunk_logs_np.append(np.array([0.0]))
                chunk_logs_np = chunk_logs_np[:event_num]

                # Check if all event types are empty
                total_events = sum(len(knots) for knots in chunk_logs_np)
                if total_events == 0:
                    baseline = np.zeros(event_num)
                    results.append(
                        {
                            "success": True,
                            "baseline": serialize_array(baseline),
                            "method": "fallback_empty",
                        }
                    )
                    continue

                # Apply Hawkes modeling
                try:
                    model = HawkesADM4(decay)
                    model.fit(
                        chunk_logs_np,
                        end_time,
                        baseline_start=np.ones(event_num) * ini_intensity,
                    )
                    baseline = np.array(model.baseline)

                    results.append(
                        {
                            "success": True,
                            "baseline": serialize_array(baseline),
                            "method": "hawkes",
                        }
                    )

                except Exception as hawkes_error:
                    # Fallback: simple counting
                    baseline = np.zeros(event_num)
                    for j, knots in enumerate(chunk_logs_np):
                        if len(knots) > 0:
                            baseline[j] = len(knots) / end_time

                    results.append(
                        {
                            "success": True,
                            "baseline": serialize_array(baseline),
                            "method": "fallback_counting",
                            "hawkes_error": str(hawkes_error),
                        }
                    )

            except Exception as batch_error:
                results.append(
                    {"success": False, "error": f"Batch {i}: {str(batch_error)}"}
                )

        return jsonify({"success": True, "results": results})

    except Exception as e:
        logger.error(f"Error in batch hawkes_modeling: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting Hawkes Process Modeling Server...")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /hawkes/model - Single chunk modeling")
    logger.info("  POST /hawkes/batch - Batch chunk modeling")

    app.run(host="0.0.0.0", port=8080, debug=False)
