#!/usr/bin/env python3
import json
import sys

d = json.load(open(sys.argv[1] + "/metadata.json"))
p = sorted(d["encodings"]["G_withdrawn"]["planes"], key=lambda q: q["z_um"])
k, n = int(sys.argv[2]), len(p)
i, j = (k, k + 1) if k < n - 1 else (k - (n - 1), k - (n - 1) + 2)
print(p[i]["section_id"].split("-")[-1] + "," + p[j]["section_id"].split("-")[-1])
