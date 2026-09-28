import json
from pathlib import Path


def read_frame(path):
    frame = json.loads(Path(path).read_text())
    if frame.get("status") != "HEADLINE":
        raise ValueError(f"{path}: evaluation frame status must be HEADLINE")
    rows = frame.get("slides")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: evaluation frame requires a nonempty slides list")
    samples = []
    for row in rows:
        if any(not isinstance(row.get(key), str) or not row[key] for key in ("sample", "group")):
            raise ValueError("frame rows require nonempty sample and group strings")
        if row.get("panel") not in ("5k", "477") or type(row.get("fold")) is not int or row["fold"] < 0:
            raise ValueError("frame rows require panel 5k/477 and a nonnegative integer fold")
        samples.append(row["sample"])
    if len(samples) != len(set(samples)):
        raise ValueError(f"{path}: duplicate sample in evaluation frame")
    return frame


def check_probability_source(npz, expected_experiment_id, path):
    if "experiment_id" not in npz.files:
        raise ValueError(f"{path}: Cell OOF array has no experiment_id")
    found = str(npz["experiment_id"])
    if found != expected_experiment_id:
        raise ValueError(f"{path}: Cell OOF experiment_id {found} differs from the frame's "
                         f"probability source {expected_experiment_id}")
