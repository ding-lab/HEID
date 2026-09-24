#!/usr/bin/env python
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
a = json.loads((out / "cells_metadata_G_withdrawn.json").read_text())
b = json.loads((out / "cells_metadata_G_not_withdrawn.json").read_text())
for k in ("canvas_px", "canvas_mm", "indices_with"):
    assert a[k] == b[k], k


for ca, cb in zip(a["classes"], b["classes"]):
    for k in ("name", "colour", "n_cells"):
        assert ca[k] == cb[k], (ca["name"], k)
    ca["norm_density_full_alpha_by_basis"] = {
        "G_withdrawn": ca.pop("norm_density_full_alpha"),
        "G_not_withdrawn": cb["norm_density_full_alpha"]}
a["planes"] = a["planes"] + b["planes"]
a["checks"] = a.get("checks", []) + b.get("checks", [])
(out / "cells_metadata.json").write_text(json.dumps(a, indent=1))
print("merged ->", out / "cells_metadata.json", len(a["planes"]), "textures")
