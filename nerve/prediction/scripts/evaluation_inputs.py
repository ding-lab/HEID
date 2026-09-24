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


def read_folds(path):
    folds = json.loads(Path(path).read_text())
    rows = folds.get("per_sample")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: fold assignments require a nonempty per_sample list")
    mapping = {}
    for row in rows:
        sample, fold = row.get("sample"), row.get("fold_he_safe")
        if not isinstance(sample, str) or not sample or type(fold) is not int or fold < 0:
            raise ValueError("fold rows require sample and a nonnegative integer fold_he_safe")
        if sample in mapping:
            raise ValueError(f"duplicate fold assignment: {sample}")
        mapping[sample] = fold
    return folds, mapping


def read_replay(path, fold_of):
    replay = json.loads(Path(path).read_text())
    if replay.get("arm") != "CELL":
        raise ValueError(f"{path}: operating-point transfer requires a CELL replay")
    rows = replay.get("per_sample")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: replay requires a nonempty per_sample list")
    seen = set()
    for row in rows:
        sample = row.get("sample")
        if sample in seen or sample not in fold_of:
            raise ValueError(f"missing or duplicate sample fold: {sample}")
        seen.add(sample)
        if not isinstance(row.get("per_tau"), dict) or not row["per_tau"]:
            raise ValueError(f"missing operating-point counts: {sample}")
        for counts in row["per_tau"].values():
            if counts.get("skipped_degenerate"):
                continue
            if any(type(counts.get(k)) is not int or counts[k] < 0 for k in ("iou_tp", "n_pred_scored", "n_gold")):
                raise ValueError(f"invalid operating-point counts: {sample}")
            if counts["iou_tp"] > min(counts["n_pred_scored"], counts["n_gold"]):
                raise ValueError(f"true-positive count exceeds prediction or reference count: {sample}")
    return replay


def check_probability_source(npz, expected_experiment_id, path):
    if "experiment_id" not in npz.files:
        raise ValueError(f"{path}: Cell OOF array has no experiment_id")
    found = str(npz["experiment_id"])
    if found != expected_experiment_id:
        raise ValueError(f"{path}: Cell OOF experiment_id {found} differs from the frame's "
                         f"probability source {expected_experiment_id}")
