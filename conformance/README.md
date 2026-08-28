# Conformance corpus

Shared test fixtures for the Python `zarr-metadata` package (the reference
implementation) and the TypeScript `zarr-metadata` port. Both test suites run
every case in this directory; a spec change updates the corpus first, turning
both implementations red until each is fixed.

## Format

One JSON file per document kind (`v3_array.json`, `v3_group.json`,
`v2_array.json`, `v2_group.json`, `v2_consolidated.json`), each an array of
cases:

```json
{
  "description": "what this case exercises",
  "document": { "the": "metadata document under test" },
  "problems": [
    { "loc": ["codecs", 0, "name"], "kind": "invalid_type" }
  ]
}
```

- An empty `problems` array means the document is structurally valid.
- Expected problems assert on `loc` (path from document root) and `kind`
  (`missing_key` | `invalid_type` | `invalid_value` | `invalid_json`) only.
  Human-readable messages are deliberately NOT asserted: they legitimately
  differ between languages (Python `repr` vs JS formatting).
- Comparison is order-insensitive: implementations may emit problems in any
  order, but must emit exactly this multiset.

## Constraints on fixtures

- Documents must be expressible as JSON text (the corpus is JSON), so cases
  exercising non-JSON runtime values (functions, NaN floats, non-string keys)
  belong in each implementation's own unit tests, not here.
- No fixture may hinge on JSON int-vs-float spelling (`3` vs `3.0`):
  `JSON.parse` collapses both to the same number in JavaScript, while Python
  distinguishes them. Python rejects `zarr_format: 3.0`; JS structurally
  cannot.

## Runners

- TypeScript: `packages/zarr-metadata-ts/test/conformance.test.ts` (vitest).
- Python: `scripts/check_conformance.py` (run against an installed or local
  `zarr-metadata`).
