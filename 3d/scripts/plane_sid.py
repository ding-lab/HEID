#!/usr/bin/env python3
import json
import sys

if len(sys.argv) > 3 and sys.argv[3] == "swap":
    d = json.load(open(sys.argv[1] + "/he_swap_metadata.json"))
    p = sorted({q["section_id"] for q in d["planes"] if q["basis"] == "G_withdrawn"})
    print(p[int(sys.argv[2])])
else:
    d = json.load(open(sys.argv[1] + "/metadata.json"))
    print(d["encodings"]["G_withdrawn"]["planes"][int(sys.argv[2])]["section_id"])
