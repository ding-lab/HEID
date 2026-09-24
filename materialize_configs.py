#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXTENSIONS = {".json", ".csv", ".tsv"}
PLACEHOLDERS = ("${PROJECTS_ROOT}", "${RELEASE_ROOT}")
SKIPPED_DIRECTORIES = {".git", ".venv", "venv", "__pycache__"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace the PROJECTS_ROOT and RELEASE_ROOT placeholders in shipped config files")
    parser.add_argument("--projects-root", required=True, help="root of the data plane")
    parser.add_argument("--release-root", default=None, help="root of this repository (default: directory containing this script)")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    release_root = Path(arguments.release_root or Path(__file__).resolve().parent).resolve()
    projects_root = Path(arguments.projects_root).resolve()
    mapping = {"${PROJECTS_ROOT}": str(projects_root), "${RELEASE_ROOT}": str(release_root)}
    changed = 0
    for path in sorted(release_root.rglob("*")):
        if not path.is_file() or path.suffix not in EXTENSIONS or SKIPPED_DIRECTORIES & set(path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not any(p in text for p in PLACEHOLDERS):
            continue
        new = text
        for key, value in mapping.items():
            new = new.replace(key, value)
        changed += 1
        print(f"{'would rewrite' if arguments.dry_run else 'rewrote'} {path.relative_to(release_root)}")
        if not arguments.dry_run:
            path.write_text(new, encoding="utf-8")
    print(f"{changed} file(s) {'to rewrite' if arguments.dry_run else 'rewritten'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
