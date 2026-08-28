"""Run the shared conformance corpus against the Python reference implementation.

The corpus (conformance/*.json) is the contract between the Python
`zarr-metadata` package and the TypeScript port; this script verifies the
Python side. The TypeScript side runs the same corpus via vitest
(test/conformance.test.ts).

Usage (with zarr-metadata importable, e.g. installed or via `uv run --with`):

    uv run --with zarr-metadata python scripts/check_conformance.py

Until zarr-metadata is published to PyPI (or to test unreleased changes),
local package instead:

    uv run --with /path/to/zarr-python/packages/zarr-metadata python scripts/check_conformance.py

Exits non-zero on any mismatch, printing one line per failing case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from zarr_metadata.model import (
    MetadataValidationError,
    ZarrV2ConsolidatedMetadata,
    validate_array_metadata_v2,
    validate_array_metadata_v3,
    validate_group_metadata_v2,
    validate_group_metadata_v3,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "conformance"


def validate_consolidated_v2(document: object) -> list[tuple[tuple[str | int, ...], str]]:
    """The v2 consolidated validator lives inside `from_json`; adapt it."""
    try:
        ZarrV2ConsolidatedMetadata.from_json(document)
    except MetadataValidationError as exc:
        return [(problem.loc, problem.kind) for problem in exc.problems]
    return []


VALIDATORS = {
    "v3_array": lambda doc: [(p.loc, p.kind) for p in validate_array_metadata_v3(doc)],
    "v3_group": lambda doc: [(p.loc, p.kind) for p in validate_group_metadata_v3(doc)],
    "v2_array": lambda doc: [(p.loc, p.kind) for p in validate_array_metadata_v2(doc)],
    "v2_group": lambda doc: [(p.loc, p.kind) for p in validate_group_metadata_v2(doc)],
    "v2_consolidated": validate_consolidated_v2,
}


def canonical(problems: list[tuple[tuple[str | int, ...], str]]) -> list[str]:
    """Order-insensitive canonical form: sorted (loc, kind) pairs."""
    return sorted(json.dumps([list(loc), kind]) for loc, kind in problems)


def main() -> int:
    corpus_kinds = sorted(path.stem for path in CORPUS_DIR.glob("*.json"))
    if corpus_kinds != sorted(VALIDATORS):
        print(f"corpus kinds {corpus_kinds} != registered validators {sorted(VALIDATORS)}")
        return 1
    failures = 0
    total = 0
    for kind, validator in VALIDATORS.items():
        cases = json.loads((CORPUS_DIR / f"{kind}.json").read_text())
        # The corpus is the cross-language contract; a truncated fixture file
        # must fail loudly, not weaken the contract silently.
        valid_count = sum(1 for case in cases if not case["problems"])
        invalid_count = len(cases) - valid_count
        if valid_count == 0 or invalid_count == 0:
            failures += 1
            print(
                f"FAIL [{kind}] corpus must hold at least one valid and one invalid case "
                f"(valid={valid_count}, invalid={invalid_count})"
            )
        for case in cases:
            total += 1
            expected = canonical(
                [(tuple(problem["loc"]), problem["kind"]) for problem in case["problems"]]
            )
            actual = canonical(validator(case["document"]))
            if actual != expected:
                failures += 1
                print(f"FAIL [{kind}] {case['description']}")
                print(f"  expected: {expected}")
                print(f"  actual:   {actual}")
    print(f"{total - failures}/{total} conformance cases passed (python)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
