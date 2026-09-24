#!/usr/bin/env python
from __future__ import annotations

import numpy as np


REACH_CUT_UM = 320.0


CORRESPONDENCE_FRAME_WHITELIST = ("fixed_original",)

_EPS = 1e-12


def log_precompensation(area_fixed_um2: float, area_moving_um2: float) -> float:
    if not (area_fixed_um2 > 0 and area_moving_um2 > 0):
        raise ValueError("footprint areas must be positive and measured, "
                         f"got fixed={area_fixed_um2} moving={area_moving_um2}")
    return 0.5 * float(np.log(area_fixed_um2 / area_moving_um2))


def precompensation_affine(log_g: float, center=None) -> np.ndarray:
    s = float(np.exp(log_g))
    G = np.array([[s, 0.0, 0.0], [0.0, s, 0.0]], dtype=np.float64)
    if center is not None:
        cx, cy = float(center[0]), float(center[1])
        G[0, 2] = (1.0 - s) * cx
        G[1, 2] = (1.0 - s) * cy
    return G


def assert_closure_neutral(log_g_ij: float, log_g_jk: float,
                           log_g_ki: float, tol: float = 1e-12) -> dict:
    resid = float(log_g_ij + log_g_jk + log_g_ki)
    if abs(resid) > tol:
        raise AssertionError(
            f"G is not closure-neutral on this triangle: residual {resid:.3e} "
            f"> {tol:.1e}.  Either the two edges disagree about a section's "
            f"area, or one edge was composed with fixed/moving swapped.")
    return {"closure_residual": resid, "tol": tol}


def apply_G(points_um: np.ndarray, log_g: float) -> np.ndarray:
    return np.asarray(points_um, dtype=np.float64) * float(np.exp(log_g))


def withdraw_G(points_um: np.ndarray, log_g: float) -> np.ndarray:
    return np.asarray(points_um, dtype=np.float64) * float(np.exp(-log_g))


def reach_um(log_g: float, r_edge_um: float) -> float:
    return abs(float(log_g)) * float(r_edge_um)


def classify_edge(log_g: float, r_edge_um: float, cut_um: float = REACH_CUT_UM) -> str:
    return "A" if reach_um(log_g, r_edge_um) <= cut_um else "B"


def assert_roundtrip(points_um: np.ndarray, log_g: float, tol: float = 1e-9) -> dict:
    x = np.asarray(points_um, dtype=np.float64)
    back = withdraw_G(apply_G(x, log_g), log_g)
    err = float(np.abs(back - x).max()) if x.size else 0.0
    scale = float(np.abs(x).max()) if x.size else 1.0
    rel = err / max(scale, _EPS)
    if rel > tol:
        raise AssertionError(f"G is not invertible to tolerance: relative "
                             f"round-trip error {rel:.3e} > {tol:.1e} "
                             f"(log_g={log_g:.6f})")
    return {"roundtrip_rel_err": rel, "tol": tol, "log_g": float(log_g)}


def assert_roundtrip_with_negative_control(points_um: np.ndarray,
                                           log_g: float) -> dict:
    ok = assert_roundtrip(points_um, log_g)
    x = np.asarray(points_um, dtype=np.float64)
    planted = withdraw_G(apply_G(x, log_g), log_g * 0.5)
    caught = float(np.abs(planted - x).max()) / max(float(np.abs(x).max()), _EPS) > 1e-9
    if not caught:
        raise AssertionError("negative control did not fire: a wrong withdrawal "
                             "constant was not detected, so the round-trip "
                             "assertion is vacuous here")
    ok["negative_control_fired"] = True
    return ok


def assert_correspondence_frame(points_into_fit: np.ndarray,
                                points_as_measured: np.ndarray,
                                log_g: float,
                                declared_frame: str) -> dict:
    a = np.asarray(points_into_fit, dtype=np.float64)
    b = np.asarray(points_as_measured, dtype=np.float64)
    if declared_frame not in CORRESPONDENCE_FRAME_WHITELIST:
        raise AssertionError(
            f"correspondences declared in frame {declared_frame!r}, which is not "
            f"whitelisted {CORRESPONDENCE_FRAME_WHITELIST}.  Feeding the fit "
            f"post-G coordinates propagates G's error into the final geometry, "
            f"where it must clear the 60 um boundary gate rather than the tile "
            f"search window.")
    if a.shape != b.shape:
        raise AssertionError(f"correspondence arrays differ in shape: "
                             f"{a.shape} into the fit vs {b.shape} as measured")
    out = {"correspondence_frame": declared_frame, "n_points": int(a.shape[0]),
           "log_g": float(log_g)}
    if a.size == 0:
        out.update(state="UNDETERMINED", reason="no correspondences")
        return out
    if log_g == 0.0:
        out.update(state="UNDETERMINED",
                   reason="log_g is exactly 0; the two frames coincide, so this "
                          "edge cannot distinguish them (and nothing is at risk)")
        return out

    dev = float(np.abs(a - b).max())
    scale = max(float(np.abs(b).max()), _EPS)
    if dev / scale > 1e-12:
        implied = float(np.median(a[b != 0] / b[b != 0])) if np.any(b != 0) else np.nan
        raise AssertionError(
            f"correspondences reaching the fit are not the measured ones: max "
            f"deviation {dev:.6g} um (relative {dev / scale:.3e}), implied scale "
            f"factor {implied:.6f} vs exp(log_g)={np.exp(log_g):.6f}.  They are "
            f"in post-G coordinates; G's error would enter the final geometry.")
    out.update(state="PASS", max_deviation_um=dev,
               reason="element-for-element identical to the measured array")
    return out


def assert_correspondence_frame_with_negative_control(
        points_into_fit: np.ndarray, points_as_measured: np.ndarray,
        log_g: float, declared_frame: str) -> dict:
    ok = assert_correspondence_frame(points_into_fit, points_as_measured, log_g,
                                     declared_frame)
    if ok["state"] != "PASS":
        ok["negative_control_fired"] = None
        return ok
    fired = False
    try:
        assert_correspondence_frame(apply_G(points_as_measured, log_g),
                                    points_as_measured, log_g, declared_frame)
    except AssertionError:
        fired = True
    if not fired:
        raise AssertionError("negative control did not fire: post-G coordinates "
                             "were not detected, so the frame assertion is "
                             "vacuous here")
    ok["negative_control_fired"] = True
    return ok


def direction_anchor(iso_values, z_fixed, z_moving) -> dict:
    iso = np.asarray(iso_values, dtype=np.float64)
    zf = np.asarray(z_fixed, dtype=np.float64)
    zm = np.asarray(z_moving, dtype=np.float64)
    if not (iso.shape == zf.shape == zm.shape):
        raise ValueError("iso / z_fixed / z_moving must have the same shape")
    return {
        "convention": "M maps moving->fixed",
        "check": "iso>0 means the moving section was magnified",
        "observed_frac_moving_deeper": float(np.mean(zm > zf)) if iso.size else None,
        "observed_median_iso": float(np.median(iso)) if iso.size else None,
        "n": int(iso.size),
        "how_to_verify": ("recompute median(0.5*log|det M|) from M_moving_to_fixed "
                          "and compare; if the sign disagrees the convention in "
                          "the writer changed and every downstream sign is suspect"),
    }


def assert_direction_anchor(anchor: dict, expect_frac_moving_deeper: float = 1.0,
                            tol: float = 1e-6) -> dict:
    got = anchor.get("observed_frac_moving_deeper")
    if got is None:
        raise AssertionError("direction anchor has no observable in it")
    if abs(got - expect_frac_moving_deeper) > tol:
        raise AssertionError(
            f"edge orientation changed: frac(moving deeper) = {got:.4f}, "
            f"expected {expect_frac_moving_deeper}.  Every sign convention "
            f"downstream (iso, k_chain, log G) is keyed to this.")
    return anchor


def _selftest() -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform(-4000, 4000, size=(500, 2))
    lg = log_precompensation(1.20e7, 1.00e7)
    assert abs(lg - 0.5 * np.log(1.2)) < 1e-12

    r = assert_roundtrip_with_negative_control(pts, lg)
    assert r["negative_control_fired"]

    r2 = assert_correspondence_frame_with_negative_control(
        pts, pts, lg, "fixed_original")
    assert r2["state"] == "PASS" and r2["negative_control_fired"], r2


    tiny = 320.0 / 3686.0 * 1e-3
    r3 = assert_correspondence_frame_with_negative_control(
        pts, pts, tiny, "fixed_original")
    assert r3["state"] == "PASS" and r3["negative_control_fired"], r3


    zero = assert_correspondence_frame(pts, pts, 0.0, "fixed_original")
    assert zero["state"] == "UNDETERMINED", zero


    try:
        assert_correspondence_frame(apply_G(pts, lg), pts, lg, "fixed_original")
        raise SystemExit("FAIL: post-G coordinates were not caught")
    except AssertionError:
        pass


    try:
        assert_correspondence_frame(pts, pts, lg, "post_G")
        raise SystemExit("FAIL: non-whitelisted frame accepted")
    except AssertionError:
        pass

    a = direction_anchor([0.017, 0.02, 0.01], [0, 10, 20], [40, 50, 60])
    assert_direction_anchor(a)
    try:
        assert_direction_anchor(direction_anchor([0.01], [50], [10]))
        raise SystemExit("FAIL: flipped orientation not caught")
    except AssertionError:
        pass

    assert classify_edge(0.01, 3686.0) == "A"
    assert classify_edge(0.20, 3686.0) == "B"


    G = precompensation_affine(lg, center=(1234.5, -678.9))
    c = np.array([1234.5, -678.9, 1.0])
    assert np.abs(G @ c - c[:2]).max() < 1e-9


    a_i, a_j, a_k = 1.20e7, 1.00e7, 0.83e7
    assert_closure_neutral(log_precompensation(a_i, a_j),
                           log_precompensation(a_j, a_k),
                           log_precompensation(a_k, a_i))
    try:
        assert_closure_neutral(log_precompensation(a_i, a_j),
                               log_precompensation(a_j, a_k),
                               log_precompensation(a_i, a_k))
        raise SystemExit("FAIL: a reversed edge passed closure neutrality")
    except AssertionError:
        pass
    print("precompensation selftest OK  "
          "(roundtrip + frame + direction, each with its negative control)")


if __name__ == "__main__":
    _selftest()
