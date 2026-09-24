
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiment_config import load_experiment

ROOT = Path(__file__).resolve().parents[1]
ARM = "native_rgb"


POOL = {
    "training": ("phikon2_vit_fpn_lr1", "experiment.json", "model_candidates.json"),
}
SELECTION_BASIS = (
    "architecture fixed by inner validation (inner validation, 5/5 folds); "
    "the architecture is fixed for this cohort"
)
CONTROLS = {
    }


def canonical_sha256(path: Path) -> str:
    payload = load_experiment(path)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_entry(tag: str, spec: tuple[str, str, str], fold: int) -> dict[str, Any] | None:
    candidate, experiment, candidates = spec
    tune = ROOT / f"outputs/{tag}/models/{candidate}/{ARM}/fold{fold}/tune"
    if not (tune / "summary.json").is_file() or not (tune / "SUCCESS.json").is_file():
        return None
    summary = json.loads((tune / "summary.json").read_text(encoding="utf-8"))
    success = json.loads((tune / "SUCCESS.json").read_text(encoding="utf-8"))
    if success.get("status") != "COMPLETED":
        return None
    experiment_path = ROOT / "configs" / experiment
    candidates_path = ROOT / "configs" / candidates


    if summary["experiment_sha256"] != canonical_sha256(experiment_path):
        raise ValueError(f"{tag} fold{fold}: tune receipt does not match {experiment}")
    if summary["candidates_sha256"] != canonical_sha256(candidates_path):
        raise ValueError(f"{tag} fold{fold}: tune receipt does not match {candidates}")
    return {
        "tag": tag,
        "candidate": candidate,
        "arm": ARM,
        "experiment_config": str(experiment_path.relative_to(ROOT)),
        "candidates_config": str(candidates_path.relative_to(ROOT)),
        "experiment_sha256": summary["experiment_sha256"],
        "candidates_sha256": summary["candidates_sha256"],
        "split_sha256": summary["split_sha256"],
        "inner_validation_score": float(summary["inner_validation_score"]),
        "selected_threshold": float(summary["selected_threshold"]),
        "best_epoch": int(summary["best_epoch"]),
        "epochs_completed": int(summary["epochs_completed"]),
        "tune_root": str(tune.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    fold = arguments.fold
    if fold not in range(5):
        raise ValueError("fold must be in [0,4]")

    pool = [entry for tag, spec in POOL.items() if (entry := read_entry(tag, spec, fold))]
    if len(pool) != len(POOL):
        missing = set(POOL) - {entry["tag"] for entry in pool}
        raise ValueError(f"fold {fold}: no completed tune for {sorted(missing)}")

    splits = {entry["split_sha256"] for entry in pool}
    if len(splits) != 1:
        raise ValueError(f"fold {fold}: candidates scored different splits {splits}")
    budgets = {entry["epochs_completed"] for entry in pool}
    if len(budgets) != 1:
        raise ValueError(f"fold {fold}: unequal epoch budgets {budgets}; not a fair comparison")

    winner = max(pool, key=lambda entry: entry["inner_validation_score"])
    others = [entry for entry in pool if entry is not winner]
    runner_up = max(others, key=lambda entry: entry["inner_validation_score"]) if others else None
    controls = [entry for tag, spec in CONTROLS.items() if (entry := read_entry(tag, spec, fold))]

    selection = {
        "schema_version": "v3-selection-1",
        "outer_fold": fold,
        "basis": "inner validation only; outer test never read",
        "split_sha256": winner["split_sha256"],
        "epochs_completed": winner["epochs_completed"],
        "selected": winner,
        "selection_basis": SELECTION_BASIS,
        "margin_over_runner_up": (
            winner["inner_validation_score"] - runner_up["inner_validation_score"]
            if runner_up else None
        ),
        "considered": sorted(pool, key=lambda entry: -entry["inner_validation_score"]),
        "controls_not_in_pool": controls,
    }


    selection["selection_payload_sha256"] = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()

    output = arguments.output or (
        ROOT / f"outputs/final/manifests/selections/fold{fold}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"fold{fold}: selected {winner['candidate']} ({winner['tag']}) "
        f"inner={winner['inner_validation_score']:.4f} thr={winner['selected_threshold']} "
        f"epoch={winner['best_epoch']}/{winner['epochs_completed']} "
        + (f"margin={selection['margin_over_runner_up']:+.4f} over {runner_up['candidate']}"
           if runner_up else "(single-candidate pool; fixed architecture)")
    )
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
