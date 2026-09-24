#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from experiment_config import load_experiment


INPUT_PATH_FIELDS = {
    'a4_data_contract', 'a4_feature_manifest', 'boundary_log_root', 'boundary_root',
    'boundary_source', 'cell_table_root', 'v4_auc_result', 'v4_split',
    'v4_training_source', 'v4_verified_result',
}
OUTPUT_PATH_FIELDS = {
    'data_contract', 'label_dataset_contract', 'label_manifest', 'labels', 'logs',
    'manifests', 'models', 'predictions', 'results', 'scratch', 'source_manifest',
}


DESCRIPTIVE_TARGETS = {
    ('experiment_id',): 'tb_territory_prediction',
    ('label_transfer', 'coordinate_source'): 'corrected_aligned_cell_table_he_px_he_py',
    ('cohort', 'low_quality_alignment_policy'): 'exclude_failed_alignment_sections',
}


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f'{path}: expected JSON object')
    return payload


def validate_relocation(original: dict, target: dict) -> list[str]:
    changes: list[str] = []

    def compare(left: Any, right: Any, key: tuple[str, ...] = ()) -> None:
        if left == right:
            return
        if len(key) == 2 and (
            key[0] == 'inputs' and key[1] in INPUT_PATH_FIELDS
            or key[0] == 'outputs' and key[1] in OUTPUT_PATH_FIELDS
        ):
            if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
                raise ValueError(f'{key}: artifact paths must be nonempty strings')
            changes.append('.'.join(key))
            return
        if key in DESCRIPTIVE_TARGETS and right == DESCRIPTIVE_TARGETS[key] and isinstance(left, str):
            changes.append('.'.join(key))
            return
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                raise ValueError(f'{key}: configuration keys differ')
            for name in left:
                compare(left[name], right[name], key + (name,))
            return
        raise ValueError(f'{".".join(key)}: scientific configuration differs')

    compare(original, target)
    return sorted(changes)


def relocate_payload(payload: dict, original: dict, target: dict,
                     original_candidates: dict, target_candidates: dict) -> tuple[dict, dict]:
    source_hash = canonical_sha256(original)
    candidate_hash = canonical_sha256(original_candidates)
    if payload.get('experiment_sha256') != source_hash:
        raise ValueError('original experiment does not match the checkpoint hash')
    if payload.get('candidates_sha256') != candidate_hash:
        raise ValueError('original candidates do not match the checkpoint hash')
    if original_candidates != target_candidates:
        raise ValueError('candidate configuration differs; relocation cannot change a model or runtime')
    changes = validate_relocation(original, target)
    candidate = payload.get('candidate_name')
    if payload.get('candidate') != target_candidates.get('candidates', {}).get(candidate):
        raise ValueError('checkpoint embedded candidate disagrees with verified candidates')
    if not payload.get('split_sha256'):
        raise ValueError('checkpoint lacks a split identity')
    if 'run_contract' in payload and payload.get('run_contract_sha256') != canonical_sha256(payload['run_contract']):
        raise ValueError('checkpoint run contract hash differs')
    migrated = dict(payload)
    migrated['experiment_sha256'] = canonical_sha256(target)
    migrated['candidates_sha256'] = canonical_sha256(target_candidates)
    receipt = {
        'operation': 'validated_configuration_relocation',
        'original_experiment_sha256': source_hash,
        'target_experiment_sha256': migrated['experiment_sha256'],
        'candidates_sha256': candidate_hash,
        'changed_fields': changes,
        'split_sha256': payload['split_sha256'],
        'weights_and_scientific_state_changed': False,
        'performance_revalidated': False,
    }
    migrated['configuration_relocation'] = receipt
    return migrated, receipt


def relocate_summary(summary: dict, payload: dict, receipt: dict) -> dict:
    expected = {
        'experiment_sha256': receipt['original_experiment_sha256'],
        'candidates_sha256': receipt['candidates_sha256'],
        'split_sha256': payload['split_sha256'],
        'arm': payload['arm'],
        'candidate': payload['candidate_name'],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f'tuning summary {key} does not match the verified checkpoint')
    if summary.get('phase') != 'tune' or not 0 <= float(summary['selected_threshold']) <= 1:
        raise ValueError('expected a completed tuning summary with a valid threshold')
    result = dict(summary)
    result['experiment_sha256'] = receipt['target_experiment_sha256']
    result['configuration_relocation'] = receipt
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'original-config', 'original-candidates', 'target-config',
                 'target-candidates', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--summary', type=Path, help='matching tuning summary used for evaluation')
    parser.add_argument('--output-summary', type=Path)
    args = parser.parse_args()
    if (args.summary is None) != (args.output_summary is None):
        parser.error('--summary and --output-summary must be supplied together')
    outputs = [args.output, args.output.with_suffix(args.output.suffix + '.relocation.json')]
    if args.output_summary:
        outputs.append(args.output_summary)
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError('output paths must be distinct')
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'refusing to replace {path}')
    original, original_candidates, target_candidates = [read_json(p) for p in
        [args.original_config, args.original_candidates, args.target_candidates]]
    target = load_experiment(args.target_config)

    import torch
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    migrated, receipt = relocate_payload(payload, original, target, original_candidates, target_candidates)
    receipt['original_checkpoint_sha256'] = file_sha256(args.checkpoint)
    migrated_summary = relocate_summary(read_json(args.summary), payload, receipt) if args.summary else None
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        torch.save(migrated, handle)
    output_receipt = dict(receipt, output_checkpoint_sha256=file_sha256(args.output))
    with outputs[1].open('x') as handle:
        json.dump(output_receipt, handle, indent=2, sort_keys=True)
        handle.write('\n')
    if args.output_summary:
        with args.output_summary.open('x') as handle:
            json.dump(migrated_summary, handle, indent=2, sort_keys=True)
            handle.write('\n')
    print(json.dumps(output_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
