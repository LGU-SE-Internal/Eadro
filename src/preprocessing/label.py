from pathlib import Path

import json
from datetime import datetime, timedelta
import pytz
from typing import Any


def extract_rcabench_labels(
    datapack: Path, config: Any
) -> tuple[datetime, datetime, datetime, datetime, str, str]:
    label_file = config.get("label_files")

    assert label_file is not None, (
        "Label file for dataset is not defined in the config."
    )
    assert isinstance(label_file, list)
    assert "injection.json" in label_file and "env.json" in label_file

    with open(datapack / "injection.json", "r") as f:
        injection = json.load(f)

    assert (
        injection is not None
        and "ground_truth" in injection
        and "service" in injection["ground_truth"]
    ), f"Invalid injection file format in {datapack / 'injection.json'}."
    gt_services = injection["ground_truth"]["service"]

    assert "fault_type" in injection
    fault_type = injection.get("fault_type")

    with open(datapack / "env.json", "r") as f:
        env = json.load(f)

    assert (
        "TIMEZONE" in env
        and "NORMAL_START" in env
        and "NORMAL_END" in env
        and "ABNORMAL_START" in env
        and "ABNORMAL_END" in env
    ), f"Invalid env file format in {datapack / 'env.json'}."

    normal_st = datetime.fromtimestamp(int(env["NORMAL_START"]), pytz.UTC) - timedelta(
        hours=8
    )
    normal_et = datetime.fromtimestamp(int(env["NORMAL_END"]), pytz.UTC) - timedelta(
        hours=8
    )
    abnormal_st = datetime.fromtimestamp(
        int(env["ABNORMAL_START"]), pytz.UTC
    ) - timedelta(hours=8)
    abnormal_et = datetime.fromtimestamp(
        int(env["ABNORMAL_END"]), pytz.UTC
    ) - timedelta(hours=8)

    return (
        normal_st,
        normal_et,
        abnormal_st,
        abnormal_et,
        gt_services[0],
        fault_type,
    )


def extract_eadro_labels(
    datapack: Path, config: Any
) -> tuple[datetime, datetime, datetime, datetime, str, str]:
    """Extract labels for eadro dataset (sn/tt)"""
    label_file = config.get("label_files")

    assert label_file is not None, (
        "Label file for dataset is not defined in the config."
    )
    assert isinstance(label_file, list), "Label file should be a list."
    assert len(label_file) > 0, "No label file defined for datapack in the config."

    label_file_path = datapack / label_file[0]

    with open(label_file_path, "r") as f:
        label = json.load(f)

    assert isinstance(label, dict), "Label file should contain a dictionary."

    gt_service = label.get("injection_name", "")
    fault_type = label.get("fault_type", "")

    normal_st = (
        datetime.fromisoformat(label["normal_start_time"])
        if label.get("normal_start_time")
        else datetime(1970, 1, 1, 0, 0, 0)
    )
    normal_et = (
        datetime.fromisoformat(label["normal_end_time"])
        if label.get("normal_end_time")
        else datetime(1970, 1, 1, 0, 0, 0)
    )
    abnormal_st = (
        datetime.fromisoformat(label["fault_start_time"])
        if label.get("fault_start_time")
        else datetime(1970, 1, 1, 0, 0, 0)
    )
    abnormal_et = (
        datetime.fromisoformat(label["fault_end_time"])
        if label.get("fault_end_time")
        else datetime(1970, 1, 1)
    )

    return normal_st, normal_et, abnormal_st, abnormal_et, gt_service, fault_type
