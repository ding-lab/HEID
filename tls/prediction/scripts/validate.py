#!/usr/bin/env python3

from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "scripts"))
import importlib

train = importlib.import_module("train")


def main() -> None:
    slides = pd.read_csv(MODULE_ROOT / "configs" / "slides_5k.tsv", sep="\t")
    manifest = pd.read_csv(
        MODULE_ROOT / "outputs" / "results" / "alignment_manifest.tsv",
        sep="\t",
    )
    current = manifest.set_index("field")
    assert len(slides) > 0
    assert slides["gen12_class"].isin(["tls_positive", "tls_negative"]).all()
    assert (~slides["cancer"].isin(["BRCA", "CHOL"])).all()
    assert slides["field"].map(current["model_scope_usable"]).all()
    assert not slides["directory_name"].str.endswith(
        ("-weak", "-blur", "-lowqc", "-bad")
    ).any()
    assert not slides["unit_id"].duplicated().any()

    label_rows = []
    for row in slides.itertuples(index=False):
        path = train.LABELS / f"{row.unit_id}_tiles.parquet"
        assert path.is_file() and path.stat().st_size > 0
        frame = pd.read_parquet(path)
        assert frame["label"].isin([-1, 0, 1]).all()
        assert (frame.loc[frame["n_cells_tile"].eq(0), "label"] == -1).all()
        assert (
            frame.loc[frame["rejected_tls_frac"].gt(0), "label"] == -1
        ).all()
        assert not frame.duplicated(["x_px", "y_px"]).any()
        label_rows.append(frame)

    all_labels = pd.concat(label_rows, ignore_index=True)
    centers = all_labels.loc[all_labels["label"].ge(0)].reset_index(drop=True)
    folds = train.balanced_group_kfold(
        centers["cv_group"].to_numpy(),
        centers["label"].to_numpy(),
        k=train.N_OUTER_FOLDS,
    )
    assert (
        pd.DataFrame({"group": centers["cv_group"], "fold": folds})
        .groupby("group")["fold"]
        .nunique()
        .max()
        == 1
    )
    for fold in range(train.N_OUTER_FOLDS):
        outer_train = np.flatnonzero(folds != fold)
        fit, val = train.group_val_split(
            outer_train,
            centers["cv_group"].to_numpy(),
            centers["label"].to_numpy(),
            42 + fold,
        )
        assert not set(centers.loc[fit, "cv_group"]).intersection(
            set(centers.loc[val, "cv_group"])
        )


    for variant in ("baseline_groupval", "querypos_groupval"):
        torch.manual_seed(42)
        model = train.build_model(variant, d=8, n_onehot=3, device=torch.device("cpu"))
        model.eval()
        center = torch.randn(4, 8)
        onehot = torch.eye(3)[torch.tensor([0, 1, 2, 0])]
        mask = torch.zeros(4, len(train.OFFSETS))
        first = model(center, torch.randn(4, len(train.OFFSETS), 8), mask, onehot)
        second = model(center, torch.randn(4, len(train.OFFSETS), 8), mask, onehot)
        assert torch.allclose(first, second, atol=1e-7)

    training_summary = json.loads(
        (MODULE_ROOT / "outputs" / "results" / "training_data.json").read_text()
    )
    observed = training_summary["observed"]
    assert observed["units"] == len(slides)
    assert observed["tiles"] == len(all_labels)
    result = {
        "ok": True,
        "units": int(len(slides)),
        "cv_groups": int(centers["cv_group"].nunique()),
        "tiles": int(len(all_labels)),
        "center_tiles": int(len(centers)),
        "positive_tiles": int(centers["label"].sum()),
        "zero_cell_tiles_ignored": int(
            all_labels["n_cells_tile"].eq(0).sum()
        ),
        "rejected_tls_overlap_tiles_ignored": int(
            all_labels["rejected_tls_frac"].gt(0).sum()
        ),
        "empty_neighbor_regression": "passed",
    }
    output = MODULE_ROOT / "outputs" / "results" / "preflight.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
