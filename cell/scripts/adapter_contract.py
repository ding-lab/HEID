
from pathlib import Path
import json

import torch

import contract as C

_RELEASE = Path(__file__).resolve().parents[2]
FINGERPRINTS = _RELEASE / "cell/configs/adapter_fingerprints.json"


def _check_fold_fingerprint(stored: dict, fold: int, description: str) -> None:
    if not isinstance(stored, dict):
        raise ValueError(f"{description}: fold fingerprint is absent")
    record = json.loads(FINGERPRINTS.read_text())
    expected = record["folds"].get(str(fold))
    if expected is None:
        raise ValueError(f"{description}: no recorded fingerprint for fold {fold}")
    for key, value in expected.items():
        if stored.get(key) != value:
            raise ValueError(f"{description}: fold fingerprint differs on {key}")
    if stored.get("cohort_binding_sha256") not in record["accepted_cohort_binding_sha256"]:
        raise ValueError(f"{description}: cohort binding is not a recorded binding of this cohort")
    if C.sha256_file(_RELEASE / record["taxonomy_file"]) != record["taxonomy_content_sha256"]:
        raise ValueError(f"{description}: taxonomy file differs from the recorded taxonomy")


def copy_lora_state(model: torch.nn.Module, state: dict) -> None:
    parameters = dict(model.named_parameters())
    expected = {name for name in parameters if name.endswith(("lora_A", "lora_B"))}
    if not isinstance(state, dict) or not expected or set(state) != expected:
        observed = set(state) if isinstance(state, dict) else set()
        raise ValueError(
            f"LoRA tensor keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    for name in sorted(expected):
        value = state[name]
        if not torch.is_tensor(value) or value.shape != parameters[name].shape:
            raise ValueError(f"LoRA tensor shape differs for {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"LoRA tensor contains non-finite values: {name}")
    with torch.no_grad():
        for name in sorted(expected):
            parameters[name].copy_(state[name].to(
                device=parameters[name].device, dtype=parameters[name].dtype))


def load_inference_adapter(path: Path, backbone: str, fold: int) -> dict:
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": "v8.adapter_checkpoint.v1",
        "arm": "raw_rgb",
        "backbone": backbone,
        "fold": fold,
        "saved_epoch": 5,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"{path}: adapter {key} differs from {value!r}")
    fingerprint = checkpoint.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError(f"{path}: adapter fingerprint is absent")
    if C.canonical_sha256(fingerprint) != checkpoint.get("fingerprint_sha256"):
        raise ValueError(f"{path}: adapter fingerprint hash mismatch")
    for key, value in {"arm": "raw_rgb", "backbone": backbone}.items():
        if fingerprint.get(key) != value:
            raise ValueError(f"{path}: adapter fingerprint differs on {key}")
    _check_fold_fingerprint(fingerprint, fold, str(path))
    state = checkpoint.get("lora_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path}: adapter has no LoRA tensors")

    manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected_manifest = {
            "schema_version": "v8.adapter_manifest.v1", "status": "PASS",
            "arm": "raw_rgb", "backbone": backbone, "outer_fold": fold,
            "canonical_fold_key": "fold_he_safe", "excludes_fold": fold,
            "saved_epoch": 5, "checkpoint_sha256": C.sha256_file(path),
            "fingerprint_sha256": checkpoint["fingerprint_sha256"],
        }
        for key, value in expected_manifest.items():
            if manifest.get(key) != value:
                raise ValueError(f"{manifest_path}: adapter manifest differs on {key}")
        _check_fold_fingerprint(manifest.get("fold_fingerprint"), fold, str(manifest_path))
    return state
