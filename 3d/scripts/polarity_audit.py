#!/usr/bin/env python
from __future__ import annotations

import numpy as np


NCC_FLOOR_BY_MODALITY = {"he": 0.022, "codex": 0.030, "xenium": 0.020}
NCC_INFORMATIVE_FLOOR = 0.030


MIN_INFORMATIVE_EDGES = 2


NCC_MARGINAL_MULTIPLE = 2.0


def edge_polarity_state(ncc_same, floor=NCC_INFORMATIVE_FLOOR,
                        marginal_multiple=NCC_MARGINAL_MULTIPLE):
    if ncc_same is None or not np.isfinite(ncc_same):
        return "NO_SIGNAL"
    hi = marginal_multiple * floor
    if ncc_same > hi:
        return "CONSISTENT"
    if ncc_same > floor:
        return "CONSISTENT_MARGINAL"
    if ncc_same < -hi:
        return "INVERTED"
    if ncc_same < -floor:
        return "INVERTED_MARGINAL"
    return "NO_SIGNAL"


def audit_sections(edge_records, floor=NCC_INFORMATIVE_FLOOR,
                   min_informative=MIN_INFORMATIVE_EDGES):
    per = {}
    for r in edge_records:
        v = r.get("ncc_same_polarity")
        if v is None or not np.isfinite(v):
            continue
        for s in (r.get("fixed"), r.get("moving")):
            if s:
                per.setdefault(s, []).append(float(v))

    out = {}
    for s, vals in per.items():
        a = np.asarray(vals)
        informative = a[np.abs(a) >= floor]
        if len(informative) < min_informative:
            verdict = "UNDETERMINED_NO_INFORMATIVE_EDGES"
        elif (informative < 0).all():
            verdict = "INVERTED"
        elif (informative > 0).all():
            verdict = "CONSISTENT"
        else:
            verdict = "MIXED_INVESTIGATE"
        out[s] = {
            "verdict": verdict,
            "n_edges": int(len(a)), "n_informative": int(len(informative)),
            "weighted_mean_ncc": float(np.sum(a * np.abs(a)) / max(np.sum(np.abs(a)), 1e-12)),
            "min_ncc": float(a.min()), "max_ncc": float(a.max()),
        }
    n_inv = sum(1 for v in out.values() if v["verdict"] == "INVERTED")
    n_mixed = sum(1 for v in out.values() if v["verdict"] == "MIXED_INVESTIGATE")
    unverifiable = sorted(k for k, v in out.items()
                          if v["verdict"] == "UNDETERMINED_NO_INFORMATIVE_EDGES")
    return {
        "per_section": out,
        "n_sections": len(out), "n_inverted": n_inv, "n_mixed": n_mixed,
        "FAIL": bool(n_inv or n_mixed),


        "POLARITY_UNVERIFIABLE_SECTIONS": unverifiable,
        "n_polarity_unverifiable": len(unverifiable),
        "report_separately_by_stratum": ("n_polarity_unverifiable must be reported "
                                         "per depth stratum, never pooled — it "
                                         "concentrates in the deep sections"),
        "floor": floor, "floor_is_empirical": True,
        "rationale": "polarity is a per-section preprocessing property, so the "
                     "evidence pools across every edge touching that section; "
                     "untextured edges carry no polarity information and are "
                     "excluded from the pool rather than deleted from the run",
    }


def _selftest():
    ok = True


    recs = [{"fixed": "A", "moving": "B", "ncc_same_polarity": 0.103},
            {"fixed": "B", "moving": "C", "ncc_same_polarity": 0.095},
            {"fixed": "C", "moving": "D", "ncc_same_polarity": -0.0019},
            {"fixed": "D", "moving": "E", "ncc_same_polarity": -0.0049},
            {"fixed": "A", "moving": "C", "ncc_same_polarity": 0.088}]
    r = audit_sections(recs)
    good = (not r["FAIL"]) and r["per_section"]["A"]["verdict"] == "CONSISTENT" \
        and r["per_section"]["E"]["verdict"] == "UNDETERMINED_NO_INFORMATIVE_EDGES"
    print(f"  consistent data + untextured edges : FAIL={r['FAIL']} "
          f"(A={r['per_section']['A']['verdict']}, "
          f"E={r['per_section']['E']['verdict']}) -> {'PASS' if good else 'FAIL'}")
    ok &= good


    bad = [{"fixed": "D", "moving": "X", "ncc_same_polarity": -0.11},
           {"fixed": "D", "moving": "Y", "ncc_same_polarity": -0.097},
           {"fixed": "X", "moving": "Y", "ncc_same_polarity": 0.10}]
    rb = audit_sections(bad)
    caught = rb["FAIL"] and rb["per_section"]["D"]["verdict"] == "INVERTED"
    print(f"  planted inversion on section D     : caught={caught} "
          f"-> {'PASS (can fail)' if caught else 'FAIL (vacuous)'}")
    ok &= caught


    quiet = [{"fixed": "Q", "moving": "R", "ncc_same_polarity": -0.0019},
             {"fixed": "Q", "moving": "S", "ncc_same_polarity": -0.0049}]
    rq = audit_sections(quiet)
    right = (not rq["FAIL"]) and \
        rq["per_section"]["Q"]["verdict"] == "UNDETERMINED_NO_INFORMATIVE_EDGES"
    print(f"  untextured-only (the old bug)      : FAIL={rq['FAIL']} "
          f"verdict={rq['per_section']['Q']['verdict']} -> "
          f"{'PASS' if right else 'FAIL'}")
    ok &= right
    return ok


if __name__ == "__main__":
    import sys
    print("polarity audit self-check")
    good = _selftest()
    print(f"\n{'PASS' if good else 'FAIL'}")
    sys.exit(0 if good else 1)
