#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from experiment_config import REFERENCE, RELEASED, cohort_size

PATH_COLUMNS = {'he_path', 'cell_table_path', 'boundary_reference_csv', 'boundary_log', 'label_path'}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected JSON object')
    return value


def verify_self_hash(value: dict, key: str) -> None:
    body = dict(value)
    claimed = body.pop(key, None)
    if claimed != canonical_sha256(body):
        raise ValueError(f'source {key} is invalid')


def map_paths(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: map_paths(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [map_paths(item, mapping) for item in value]
    if isinstance(value, str):
        for source, target in sorted(mapping.items(), key=lambda item: -len(item[0])):
            source, target = source.rstrip('/'), target.rstrip('/')
            if value == source or value.startswith(source + '/'):
                return target + value[len(source):]
    return value


def validate_manifest(original: list[dict], target: list[dict], mapping: dict[str, str]) -> None:
    source = {str(row['sample']): row for row in original}
    staged = {str(row['sample']): row for row in target}
    if len(source) != len(original) or len(staged) != len(target) or source.keys() != staged.keys():
        raise ValueError('manifest sample identities or uniqueness differ')
    for sample, row in source.items():
        expected = {key: map_paths(value, mapping) if key in PATH_COLUMNS else value
                    for key, value in row.items()}
        if expected != staged[sample]:
            keys = sorted(key for key in set(expected) | set(staged[sample])
                          if expected.get(key) != staged[sample].get(key))
            raise ValueError(f'{sample}: manifest fields changed beyond explicit relocation: {keys}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['original-contract', 'original-reference-manifest', 'original-source-manifest',
                 'original-feature-manifest', 'original-input-receipt', 'path-map']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--replace-template', action='store_true',
                        help='replace the packaged data-contract template after all validation succeeds')
    parser.add_argument('--write-manifests', action='store_true',
                        help='write the staged cohort and reference manifests from the verified originals '
                             'through the path map when they are absent')
    args = parser.parse_args()
    import pandas as pd
    root = args.project_root.resolve()
    original = read_json(args.original_contract)
    receipt = read_json(args.original_input_receipt)
    feature = read_json(args.original_feature_manifest)
    mapping = read_json(args.path_map)
    if not mapping or any(not isinstance(k, str) or not isinstance(v, str)
                          or not Path(k).is_absolute() or not Path(v).is_absolute()
                          for k, v in mapping.items()):
        raise ValueError('path-map must map explicit absolute source prefixes to absolute target prefixes')
    verify_self_hash(original, 'contract_payload_sha256')
    verify_self_hash(receipt, 'receipt_payload_sha256')
    if receipt.get('status') != 'COHORT_LABEL_AND_HE_CONTENT_VERIFIED':
        raise ValueError('source receipt does not attest completed input-content verification')
    for name, path in [('source_manifest', args.original_source_manifest),
                       ('data_contract', args.original_contract),
                       ('a4_feature_manifest', args.original_feature_manifest)]:
        if receipt.get('artifacts', {}).get(name, {}).get('sha256') != file_sha256(path):
            raise ValueError(f'source input receipt does not bind {name}')
    if original['sources']['a4_full_content_feature_manifest']['sha256'] != file_sha256(args.original_feature_manifest):
        raise ValueError('source contract does not bind the feature attestation')
    read_rows = lambda path: pd.read_csv(path, keep_default_na=False).to_dict('records')
    reference = read_rows(args.original_reference_manifest)
    reference_size, released_size = cohort_size(REFERENCE), cohort_size(RELEASED)
    if len(reference) != reference_size['n_samples'] or canonical_sha256(reference) != original['manifest']['records_sha256']:
        raise ValueError('source reference manifest does not match its bound reference contract')
    cohort = read_rows(args.original_source_manifest)
    if len(cohort) != released_size['n_samples'] or len({r['patient'] for r in cohort}) != released_size['n_patients']:
        raise ValueError('source cohort differs from the bound section and patient counts')
    staged_reference_path = root / 'data/manifests/source_cohort_reference.csv'
    staged_source_path = root / 'data/manifests/source_cohort.csv'
    if args.write_manifests:
        for rows, path, source in [(reference, staged_reference_path, args.original_reference_manifest),
                                   (cohort, staged_source_path, args.original_source_manifest)]:
            if path.exists():
                raise FileExistsError(f'refusing to replace staged manifest {path}')
            columns = list(pd.read_csv(source, keep_default_na=False, nrows=0).columns)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{key: map_paths(row[key], mapping) if key in PATH_COLUMNS else row[key]
                           for key in columns} for row in rows]).to_csv(path, index=False)
    staged_reference, staged_cohort = read_rows(staged_reference_path), read_rows(staged_source_path)
    validate_manifest(reference, staged_reference, mapping)
    validate_manifest(cohort, staged_cohort, mapping)
    relocated_feature = map_paths(feature, mapping)
    target = map_paths(original, mapping)
    feature_path = Path(target['sources']['a4_full_content_feature_manifest']['path'])
    if feature_path.resolve() == args.original_feature_manifest.resolve():
        raise ValueError('feature output must be separate from the original input')
    feature_text = json.dumps(relocated_feature, indent=2, sort_keys=True) + '\n'
    if feature_path.exists() and feature_path.read_text() != feature_text:
        raise FileExistsError(f'refusing to replace different feature attestation {feature_path}')
    target['sources']['a4_full_content_feature_manifest']['sha256'] = hashlib.sha256(feature_text.encode()).hexdigest()
    target['manifest']['path'] = str(staged_reference_path)
    target['manifest']['records_sha256'] = canonical_sha256(staged_reference)
    target['configuration_relocation'] = {
        'source_contract_sha256': file_sha256(args.original_contract),
        'source_input_receipt_sha256': file_sha256(args.original_input_receipt),
        'source_manifest_sha256': file_sha256(args.original_source_manifest),
        'staged_manifest_sha256': file_sha256(staged_source_path),
        'scientific_and_identity_fields_unchanged': True,
        'staged_image_and_label_content_reverified': False,
        'next_step': 'export labels, finalize labels, build tile index, then audit input content',
    }
    target.pop('contract_payload_sha256', None)
    target['contract_payload_sha256'] = canonical_sha256(target)
    destination = root / 'data/manifests/data_contract.json'
    if destination.exists() and not args.replace_template:
        raise FileExistsError('data_contract.json exists; use --replace-template after checking the target workspace')
    if destination.resolve() == args.original_contract.resolve():
        raise ValueError('output must not replace the original contract')
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if not feature_path.exists():
        feature_path.write_text(feature_text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(target, indent=2, sort_keys=True) + '\n')
    print(json.dumps(target['configuration_relocation'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
