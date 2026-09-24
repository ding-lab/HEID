#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from experiment_config import load_experiment

PROJECT = Path(__file__).resolve().parents[1]
TRAIN = PROJECT / "scripts" / "train_region_model_nested.py"
SELECT = PROJECT / "scripts" / "select_fold.py"
CONFIG = PROJECT / "configs" / "experiment.json"
CANDIDATES = PROJECT / "configs" / "model_candidates.json"
BASELINE = PROJECT / "configs" / "baseline_controls" / "experiment_baseline.json"
BASELINE_CANDIDATES = PROJECT / "configs" / "baseline_controls" / "model_candidates_baseline.json"

CANDIDATE = "phikon2_vit_fpn_lr1"
ARM = "native_rgb"
TUNE_EPOCHS = 16
BATCH = 4
ACCUMULATION = 4
POSITION_TOKENS = (1024 // 16) ** 2 + 1
DETERMINISTIC_ENV = {"PYTHONHASHSEED": "24073", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(path: Path) -> str:
    text = json.dumps(load_experiment(path), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(args: list[str]) -> None:
    env = dict(os.environ, **DETERMINISTIC_ENV)
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, env=env)


def check_single_factor_contract() -> None:
    baseline, arm = read_json(BASELINE), read_json(CONFIG)
    base_candidates, arm_candidates = read_json(BASELINE_CANDIDATES), read_json(CANDIDATES)
    if baseline["training"] != arm["training"]:
        raise ValueError("the arm's training block differs from the control")
    if baseline["preprocessing"] != arm["preprocessing"]:
        raise ValueError("the arm's preprocessing block differs from the control")
    control = {k: v for k, v in base_candidates["candidates"]["pre_convnext_fpn"].items() if k != "role"}
    carried = {k: v for k, v in arm_candidates["candidates"]["pre_convnext_fpn"].items() if k != "role"}
    if control != carried:
        raise ValueError("the control candidate was altered in the arm's candidates")
    training = baseline["training"]
    if training["batch_size_per_gpu"] * training["gradient_accumulation_steps"] != BATCH * ACCUMULATION:
        raise ValueError("the control's effective batch differs from the arm's")


def check_position_tokens(checkpoint: Path) -> None:
    import torch

    weights = torch.load(checkpoint, map_location="meta", weights_only=False)["model_state_dict"]
    key = next(k for k in weights if "position_embeddings" in k)
    tokens = int(weights[key].shape[1])
    if tokens != POSITION_TOKENS:
        raise ValueError(f"{checkpoint} carries a {tokens}-token position table, expected {POSITION_TOKENS}")


def model_root(tag: str, fold: int) -> Path:
    return PROJECT / "outputs" / tag / "models" / CANDIDATE / ARM / f"fold{fold}"


def tune(fold: int, device: str, workers: int) -> None:
    check_single_factor_contract()
    root = PROJECT / read_json(CONFIG)["outputs"]["models"] / CANDIDATE / ARM / f"fold{fold}" / "tune"
    if (root / "SUCCESS.json").is_file() and (root / "SUCCESS.json").stat().st_size:
        summary = read_json(root / "summary.json")
        if summary["experiment_sha256"] != canonical_sha256(CONFIG):
            raise ValueError("completed tune has a different experiment hash")
        if summary["candidates_sha256"] != canonical_sha256(CANDIDATES):
            raise ValueError("completed tune has a different candidates hash")
        check_position_tokens(root / "best.pt")
        print(f"tune fold {fold} already complete")
    else:
        resume = ["--resume"] if (root / "latest.pt").is_file() else []
        run([str(TRAIN), "--config", str(CONFIG), "--candidates-config", str(CANDIDATES),
             "--mode", "tune", "--candidate", CANDIDATE, "--outer-fold", str(fold),
             "--device", device, "--epochs", str(TUNE_EPOCHS), "--batch-size", str(BATCH),
             "--gradient-accumulation", str(ACCUMULATION), "--workers", str(workers), *resume])
    success, summary = read_json(root / "SUCCESS.json"), read_json(root / "summary.json")
    if success["status"] != "COMPLETED":
        raise ValueError("tune did not complete")
    for payload in (success, summary):
        if payload["candidate"] != CANDIDATE or int(payload["epochs_completed"]) != TUNE_EPOCHS:
            raise ValueError("tune receipt has a different candidate or epoch count")
    score = float(summary["inner_validation_score"])
    if not (math.isfinite(score) and 0.0 <= score <= 1.0):
        raise ValueError(f"inner validation score out of range: {score}")
    print(f"tune fold {fold}: inner Dice {score:.4f}, best epoch {summary['best_epoch']}")


def refit_and_evaluate(fold: int, device: str, workers: int) -> None:
    selection_path = PROJECT / "outputs" / "final" / "manifests" / "selections" / f"fold{fold}.json"
    selection = read_json(selection_path)
    if int(selection["outer_fold"]) != fold:
        raise ValueError("selection manifest is for another fold")
    chosen = selection["selected"]
    config = PROJECT / chosen["experiment_config"]
    candidates = PROJECT / chosen["candidates_config"]
    root = model_root(chosen["tag"], fold)
    summary = read_json(root / "tune" / "summary.json")
    for field in ("experiment_sha256", "candidates_sha256", "split_sha256"):
        if summary[field] != chosen[field]:
            raise ValueError(f"tune summary disagrees with the selection on {field}")
    if int(summary["best_epoch"]) != int(chosen["best_epoch"]):
        raise ValueError("tune summary disagrees with the selection on the epoch")
    if abs(float(summary["selected_threshold"]) - float(chosen["selected_threshold"])) >= 1e-9:
        raise ValueError("tune summary disagrees with the selection on the threshold")
    if chosen["candidate"].startswith("phikon2_"):
        check_position_tokens(root / "tune" / "best.pt")

    refit = root / "refit"
    if not (refit / "SUCCESS.json").is_file():
        resume = ["--resume"] if (refit / "latest.pt").is_file() else []
        run([str(TRAIN), "--config", str(config), "--candidates-config", str(candidates),
             "--mode", "refit", "--candidate", chosen["candidate"], "--outer-fold", str(fold),
             "--device", device, "--batch-size", str(BATCH),
             "--gradient-accumulation", str(ACCUMULATION), "--workers", str(workers), *resume])
    success, refit_summary = read_json(refit / "SUCCESS.json"), read_json(refit / "summary.json")
    if success["status"] != "COMPLETED":
        raise ValueError("refit did not complete")
    for payload in (success, refit_summary):
        if payload["candidate"] != chosen["candidate"] or int(payload["outer_fold"]) != fold:
            raise ValueError("refit receipt has a different candidate or fold")
    if int(refit_summary["epochs_completed"]) != int(chosen["best_epoch"]):
        raise ValueError("refit ran a different number of epochs than the tune selected")

    run([str(TRAIN), "--config", str(config), "--candidates-config", str(candidates),
         "--mode", "evaluate", "--candidate", chosen["candidate"], "--outer-fold", str(fold),
         "--device", device, "--batch-size", str(BATCH), "--workers", str(workers),
         "--save-predictions"])
    result_path = (PROJECT / "outputs" / chosen["tag"] / "results" / chosen["candidate"] / ARM
                   / f"fold{fold}_outer_metrics.json")
    result = read_json(result_path)
    if int(result["outer_fold"]) != fold:
        raise ValueError("outer result is for another fold")
    if abs(float(result["threshold_from_inner_validation"]) - float(chosen["selected_threshold"])) >= 1e-9:
        raise ValueError("outer evaluation used a threshold the inner validation did not choose")
    dice = float(result["aggregate"]["patient_equal_tumor_dice"])
    if not (math.isfinite(dice) and 0.0 <= dice <= 1.0):
        raise ValueError(f"patient-equal Dice out of range: {dice}")
    print(f"fold {fold}: threshold {result['threshold_from_inner_validation']}, "
          f"patient-equal Dice {dice:.4f}, sections {result['aggregate']['n_samples']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--stage", choices=("all", "tune", "select", "refit"), default="all",
                        help="refit also evaluates the refitted model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.device == "cuda":
        import torch

        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            raise RuntimeError("a CUDA device with bfloat16 support is required")
    if args.stage in ("all", "tune"):
        tune(args.fold, args.device, args.workers)
    if args.stage in ("all", "select"):
        run([str(SELECT), "--fold", str(args.fold)])
    if args.stage in ("all", "refit"):
        refit_and_evaluate(args.fold, args.device, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
